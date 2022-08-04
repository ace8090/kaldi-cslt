#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Begin configuration.
stage=-4 #  This allows restarting after partway, when something when wrong.
config=
cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1" # 缩放参数
realign_iters="10 20 30"; # 对齐的迭代次数
num_iters=35    # Number of iterations of training
max_iter_inc=25 # Last iter to increase #Gauss on.
beam=10 # beam搜索的初始搜索值
careful=false # 是否使用更加严格的搜索方式
retry_beam=40 # beam搜索的重试搜索值
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
power=0.25 # Exponent for number of gaussians according to occurrence counts
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves
norm_vars=false # deprecated.  Prefer --cmvn-opts "--norm-vars=true"
                # use the option --cmvn-opts "--norm-means=false"
cmvn_opts= # cmvn参数
delta_opts= # delta参数
context_opts=   # use"--context-width=5 --central-position=2" for quinphone
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; # source the path. # 设置环境变量
. parse_options.sh || exit 1;  # 解析-name value 类型参数

# 判断除了-name参数外，其余参数个数是否等于6，如果不等于6，则报错并退出
if [ $# != 6 ]; then
   echo "Usage: steps/train_deltas.sh <num-leaves> <tot-gauss> <data-dir> <lang-dir> <alignment-dir> <exp-dir>"
   echo "e.g.: steps/train_deltas.sh 2000 10000 data/train_si84_half data/lang exp/mono_ali exp/tri1"
   echo "main options (for others, see top of script file)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --config <config-file>                           # config containing options"
   echo "  --stage <stage>                                  # stage to do partial re-run from."
   exit 1;
fi

# 参数1：叶子节点数量，参数2：总的Gauss数量，参数3：数据目录，参数4：语言模型目录，参数5：对齐目录，参数6：输出目录
numleaves=$1
totgauss=$2
data=$3
lang=$4
alidir=$5
dir=$6

# 判断 $alidir/final.mdl $alidir/ali.1.gz $data/feats.scp $lang/phones.txt 是否存在
for f in $alidir/final.mdl $alidir/ali.1.gz $data/feats.scp $lang/phones.txt; do
  [ ! -f $f ] && echo "train_deltas.sh: no such file $f" && exit 1;
done

numgauss=$numleaves # 初始化Gauss数量
incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # per-iter increment for #Gauss # 每次迭代增加的Gauss数量
oov=`cat $lang/oov.int` || exit 1; # 用于替换OOV的词语索引
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1; # 设置上下文无关的词语索引
nj=`cat $alidir/num_jobs` || exit 1; # 设置进程数量
mkdir -p $dir/log # 创建日志目录
echo $nj > $dir/num_jobs # 设置进程数量

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1; # 检查语言模型和模型源目录的语言类型是否一致
cp $lang/phones.txt $dir || exit 1; # 复制$lang/phones.txt到输出目录

sdata=$data/split$nj; # 设置分割数据目录
split_data.sh $data $nj || exit 1; # 按照线程分割数据


[ $(cat $alidir/cmvn_opts 2>/dev/null | wc -c) -gt 1 ] && [ -z "$cmvn_opts" ] && \
  echo "$0: warning: ignoring CMVN options from source directory $alidir" # 如果$alidir/cmvn_opts存在，则警告
$norm_vars && cmvn_opts="--norm-vars=true $cmvn_opts" # 设置cmvn_opts参数
echo $cmvn_opts  > $dir/cmvn_opts # keep track of options to CMVN.
[ ! -z $delta_opts ] && echo $delta_opts > $dir/delta_opts # keep track of options to delta.

# 首先从feat.scp中读取训练特征做CMVN，并写入cmvn.scp文件中；然后通过管道传递做了CMVN的数据，再进行delta，最终赋值给feats
feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |"

rm $dir/.error 2>/dev/null # 删除$dir/.error文件

# 读取特征文件及其对应的对齐信息,计算决策树聚类过程中需要的一些统计量
# 各个phone的特征均值、方差，以及该phone所出现的语音帧数量等
if [ $stage -le -3 ]; then
  echo "$0: accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    acc-tree-stats $context_opts \
    --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
    "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;
  sum-tree-stats $dir/treeacc $dir/*.treeacc 2>$dir/log/sum_tree_acc.log || exit 1;
  rm $dir/*.treeacc
fi


if [ $stage -le -2 ]; then
  # 对单音素进行了一个相似性的聚类，生成了一套音素集合，即问题集（Question set)
  echo "$0: getting questions for tree-building, via clustering"
  # preparing questions, roots file...
  cluster-phones $context_opts $dir/treeacc $lang/phones/sets.int \
    $dir/questions.int 2> $dir/log/questions.log || exit 1;
  cat $lang/phones/extra_questions.int >> $dir/questions.int
  compile-questions $context_opts $lang/topo $dir/questions.int \
    $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

  # 基于HMM 状态进行决策树的构建
  echo "$0: building the tree"
  $cmd $dir/log/build_tree.log \
    build-tree $context_opts --verbose=1 --max-leaves=$numleaves \
    --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;

  # 对模型参数进行绑定，生成初始模型
  $cmd $dir/log/init_model.log \
    gmm-init-model  --write-occs=$dir/1.occs  \
      $dir/tree $dir/treeacc $lang/topo $dir/1.mdl || exit 1;
  if grep 'no stats' $dir/log/init_model.log; then
     echo "** The warnings above about 'no stats' generally mean you have phones **"
     echo "** (or groups of phones) in your phone set that had no corresponding data. **"
     echo "** You should probably figure out whether something went wrong, **"
     echo "** or whether your data just doesn't happen to have examples of those **"
     echo "** phones. **"
  fi

  gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl 2>$dir/log/mixup.log || exit 1; # 进行混合操作
  rm $dir/treeacc # 删除treeacc文件
fi

# 将原有对齐中的相关信息进行转换,以保证与新tree文件一致
if [ $stage -le -1 ]; then
  # Convert the alignments.
  echo "$0: converting alignments from $alidir to use current tree"
  $cmd JOB=1:$nj $dir/log/convert.JOB.log \
    convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree \
     "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi
# 建立训练graph
if [ $stage -le 0 ]; then
  echo "$0: compiling graphs of transcripts"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/1.mdl  $lang/L.fst  \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text |" \
      "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

x=1 # 初始化x为1
# 模型训练：指定轮数的EM算法迭代，按照固定间隔穿插维特比重对齐和GMM模型的高斯混合分量分裂，从而获得更好的模型参数
while [ $x -lt $num_iters ]; do
  echo "$0: training pass $x"
  if [ $stage -le $x ]; then
      # 如果 $x 在 $realign_iters 中，那么进行对齐操作
    if echo $realign_iters | grep -w $x >/dev/null; then
      echo "$0: aligning data"
      mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |" # 设置模型参数
      $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
        gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
         "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
         "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
    fi
    # 计算每个高斯分布的均值和方差，每个任务输出到一个acc文件
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali  $dir/$x.mdl "$feats" \
       "ark,s,cs:gunzip -c $dir/ali.JOB.gz|" $dir/$x.JOB.acc || exit 1;
    # 根据第x次的的模型统计量，构建新模型[x+1].mdl
    $cmd $dir/log/update.$x.log \
      gmm-est --mix-up=$numgauss --power=$power \
        --write-occs=$dir/$[$x+1].occs $dir/$x.mdl \
       "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc # 删除旧模型和统计量
    rm $dir/$x.occs # 删除$dir/$x.occs
  fi
  [ $x -le $max_iter_inc ] && numgauss=$[$numgauss+$incgauss]; # 如果$x <= $max_iter_inc，那么$numgauss加上$incgauss
  x=$[$x+1]; # 增加x的值
done

rm $dir/final.mdl $dir/final.occs 2>/dev/null # 删除旧模型和统计量
ln -s $x.mdl $dir/final.mdl # 建立新模型的软链接
ln -s $x.occs $dir/final.occs # 建立$x.occs的软链接

steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir # 进行诊断操作

# Summarize warning messages...
utils/summarize_warnings.pl  $dir/log # 汇总警告信息

steps/info/gmm_dir_info.pl $dir # 输出模型相关信息

echo "$0: Done training system with delta+delta-delta features in $dir"

exit 0
