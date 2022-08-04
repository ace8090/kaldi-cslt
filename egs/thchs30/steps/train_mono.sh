#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#           2019  Xiaohui Zhang
# Apache 2.0


# To be run from ..
# Flat start and monophone training, with delta-delta features.
# This script applies cepstral mean normalization (per speaker).

# Begin configuration section.
nj=4
cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1" # 缩放参数
num_iters=40    # Number of iterations of training
max_iter_inc=30 # Last iter to increase #Gauss on.
initial_beam=6 # beam used in the first iteration (set smaller to speed up initialization)
regular_beam=10 # beam used after the first iteration
retry_beam=40
totgauss=1000 # Target #Gaussians.
careful=false
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
realign_iters="1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 23 26 29 32 35 38";
config= # name of config file.
stage=-4
power=0.25 # exponent to determine number of gaussians from occurrence counts
norm_vars=false # deprecated, prefer --cmvn-opts "--norm-vars=false"
cmvn_opts=  # can be used to add extra options to cmvn.
delta_opts= # can be used to add extra options to add-deltas
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi # 设置环境变量
. parse_options.sh || exit 1; # -name value 类型的参数解析

# 判断 -name 类型以外参数个数是否等于3，如果不等于3，则退出
if [ $# != 3 ]; then
  echo "Usage: steps/train_mono.sh [options] <data-dir> <lang-dir> <exp-dir>"
  echo " e.g.: steps/train_mono.sh data/train.1k data/lang exp/mono"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

# 参数1：训练数据目录，参数2：语言模型目录，参数3：输出目录
data=$1
lang=$2
dir=$3

oov_sym=`cat $lang/oov.int` || exit 1; # 设置界外词符号

mkdir -p $dir/log # 创建log文件夹
echo $nj > $dir/num_jobs # 将并行线程数nj写入num_jobs文件
sdata=$data/split$nj; # 设置并行目录
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1; # 按照线程数nj分割数据集

cp $lang/phones.txt $dir || exit 1; # 复制语言模型文件中的phones.txt文件到目标目录


$norm_vars && cmvn_opts="--norm-vars=true $cmvn_opts" # 设置cmvn_opts
echo $cmvn_opts  > $dir/cmvn_opts # keep track of options to CMVN. # 
[ ! -z $delta_opts ] && echo $delta_opts > $dir/delta_opts # keep track of options to delta

# 首先从feat.scp中读取训练特征做CMVN，并写入cmvn.scp文件中；然后通过管道传递做了CMVN的数据，再进行delta，最终赋值给feats
feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |" # 读取特征文件，并进行CMVN和delta
example_feats="`echo $feats | sed s/JOB/1/g`"; # 从feats变量中获取第一个JOB（线程）的特征

# monophone 系统初始化
echo "$0: Initializing monophone system."

[ ! -f $lang/phones/sets.int ] && exit 1; # 判断语言模型文件中是否有phones/sets.int文件
shared_phones_opt="--shared-phones=$lang/phones/sets.int" # 设置共享phone符号

if [ $stage -le -3 ]; then
  # Note: JOB=1 just uses the 1st part of the features-- we only need a subset anyway.
  # 利用feat-to-dim获取特征维度，如果获取失败，则退出
  if ! feat_dim=`feat-to-dim "$example_feats" - 2>/dev/null` || [ -z $feat_dim ]; then
    feat-to-dim "$example_feats" -
    echo "error getting feature dimension"
    exit 1;
  fi
  # 利用第一个子集初始化模型,构造模型文件和决策树文件，如果失败，则退出
  # 输出为 $dir/0.mdl $dir/tree
  $cmd JOB=1 $dir/log/init.log \
    gmm-init-mono $shared_phones_opt "--train-feats=$feats subset-feats --n=10 ark:- ark:-|" $lang/topo $feat_dim \
    $dir/0.mdl $dir/tree || exit 1;
fi

numgauss=`gmm-info --print-args=false $dir/0.mdl | grep gaussians | awk '{print $NF}'` # 从$dir/0.mdl获取模型中的gaussians个数
# 计算当前高斯数：（总高斯数-当前高斯数）/高斯迭代次数 得到每次迭代需要增加的高斯数
incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # per-iter increment for #Gauss

# 训练网络是将每个句子构造一个音素级别的fst网络，其中$sdata/JOB/text中是包含对每个句子的单词级别的标注，L.fst是字典的fst表示，将phones转换为words
# 构造monophone解码图就是将text的每个句子生成fst，将fst与L.fst进行结合（composition）形成训练有用的音素级别fst网络
# sts.JOB.gz采用key-value对每个句子的fst网络进行保存，并且value保存的句子是每个句子中两个音素之间互联的边，如标注a,b,c，value的保存是a->b,b->c
if [ $stage -le -2 ]; then
  echo "$0: Compiling training graphs"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/0.mdl  $lang/L.fst \
    "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $sdata/JOB/text|" \
    "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

# align-equal-compiled 对每一帧特征进行对齐操作，因为单音素模型没有对齐模型，所以采用均匀对齐方式，保存在fsts.JOB.gz
# gmm-acc-stats-ali 根据对齐信息，计算每个高斯分布的均值和方差，每个任务输出到一个acc文件
if [ $stage -le -1 ]; then
  echo "$0: Aligning data equally (pass 0)"
  $cmd JOB=1:$nj $dir/log/align.0.JOB.log \
    align-equal-compiled "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" ark,t:-  \| \
    gmm-acc-stats-ali --binary=true $dir/0.mdl "$feats" ark:- \
    $dir/0.JOB.acc || exit 1;
fi

# In the following steps, the --min-gaussian-occupancy=3 option is important, otherwise
# we fail to est "rare" phones and later on, they never align properly.
# 根据上面的统计量更新GMM模型，min-gaussian-occupancy 的作用是设置occupancy的阈值
# 如果某个单高斯Component的occupancy_低于这个阈值，那么就不会更新这个高斯
if [ $stage -le 0 ]; then
  gmm-est --min-gaussian-occupancy=3  --mix-up=$numgauss --power=$power \
    $dir/0.mdl "gmm-sum-accs - $dir/0.*.acc|" $dir/1.mdl 2> $dir/log/update.0.log || exit 1;
  rm $dir/0.*.acc
fi

beam=$initial_beam # will change to regular_beam below after 1st pass
# note: using slightly wider beams for WSJ vs. RM.
x=1 #循环变量x初始设置为1
# 循环迭代，每次迭代都会更新GMM模型，并且更新对齐信息，每次迭代都会更新对齐信息，并且更新对齐信息
while [ $x -lt $num_iters ]; do
  echo "$0: Pass $x"
  if [ $stage -le $x ]; then
    if echo $realign_iters | grep -w $x >/dev/null; then
      # 如果 $x 在 $realign_iters 中，那么进行对齐操作
      echo "$0: Aligning data"
      mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |"
      $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
        gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
        "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" \
        || exit 1;
    fi
    # 计算每个高斯分布的均值和方差，每个任务输出到一个acc文件
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali  $dir/$x.mdl "$feats" "ark:gunzip -c $dir/ali.JOB.gz|" \
      $dir/$x.JOB.acc || exit 1;

    # 根据第x次的的模型统计量，构建新模型[x+1].mdl
    $cmd $dir/log/update.$x.log \
      gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss --power=$power $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc|" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs 2>/dev/null # 删除旧的模型和统计量
  fi
  # 增加高斯个数
  if [ $x -le $max_iter_inc ]; then
     numgauss=$[$numgauss+$incgauss];
  fi
  beam=$regular_beam # 将beam设置更新
  x=$[$x+1] # x加1
done

# 设置final.mdl为最后一次迭代生成模型的软连接
( cd $dir; rm final.{mdl,occs} 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs )


steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir # 检查对齐信息是否正确
utils/summarize_warnings.pl $dir/log # 检查是否有警告

steps/info/gmm_dir_info.pl $dir # 打印模型信息

echo "$0: Done training monophone system in $dir"

exit 0

# example of showing the alignments:
# show-alignments data/lang/phones.txt $dir/30.mdl "ark:gunzip -c $dir/ali.0.gz|" | head -4

