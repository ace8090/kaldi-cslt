#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Compute training alignments using a model with delta or
# delta + delta-delta features.

# If you supply the "--use-graphs true" option, it will use the training
# graphs from the source directory (where the model is).  In this
# case the number of jobs must match with the source directory.


# Begin configuration section.
nj=4 # 并行线程数
cmd=run.pl # 运行命令
use_graphs=false # 是否使用训练模型中的图
# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1" # 缩放参数
beam=10 # beam搜索的初始搜索值
retry_beam=40 # beam搜索的重试搜索值
careful=false # 是否使用更严格的搜索方式
boost_silence=1.0 # Factor by which to boost silence during alignment. # 在对齐时slience的概率
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path. # 设置环境变量
. parse_options.sh || exit 1; # 解析-name value 类型参数

# 判断除了-name参数外，其余参数个数是否等于4，如果不等于4，则报错并退出
if [ $# != 4 ]; then
   echo "usage: steps/align_si.sh <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  steps/align_si.sh data/train data/lang exp/tri1 exp/tri1_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --use-graphs true                                # use graphs in src-dir"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

# 参数1：数据目录，参数2：语言模型目录，参数3：模型源目录，参数4：模型对齐目录
data=$1
lang=$2
srcdir=$3
dir=$4

# 判断 $data/text $lang/oov.int $srcdir/tree $srcdir/final.mdl 是否存在
for f in $data/text $lang/oov.int $srcdir/tree $srcdir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

oov=`cat $lang/oov.int` || exit 1; # 获取界外词的索引
mkdir -p $dir/log # 创建对齐日志目录
echo $nj > $dir/num_jobs # 将并行线程数写入文件
sdata=$data/split$nj # 设置并行目录
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
cp $srcdir/splice_opts $dir 2>/dev/null # frame-splicing options.
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
cp $srcdir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
delta_opts=`cat $srcdir/delta_opts 2>/dev/null`
cp $srcdir/delta_opts $dir 2>/dev/null

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1; # 按照线程数nj分割数据集

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1; # 检查语言模型和模型源目录的语言类型是否一致 
cp $lang/phones.txt $dir || exit 1; # 复制语言模型文件中的phones.txt文件到目标目录

cp $srcdir/{tree,final.mdl} $dir || exit 1; # 复制模型源目录中的tree和final.mdl文件到目标目录
cp $srcdir/final.occs $dir; # 复制模型源目录中的final.occs文件到目标目录


# 如果存在 $srcdir/final.mat 设置feat_type=lda 否则设置feat_type=delta
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

# 根据不同的feat_type获取特征并做相应的变换，得到feats
case $feat_type in
  # 首先从feat.scp中读取训练特征做CMVN，并写入cmvn.scp文件中；然后通过管道传递做了CMVN的数据，再进行delta，最终赋值给feats
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  # 首先从feat.scp中读取训练特征做CMVN，并写入cmvn.scp文件中；然后通过管道传递做了CMVN的数据，再进行LDA，最终赋值给feats
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    cp $srcdir/final.mat $srcdir/full.mat $dir # 将LDA模型复制到目标目录
   ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

echo "$0: aligning data in $data using model from $srcdir, putting alignments in $dir" # 输出日志信息

mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/final.mdl - |" # 设置模型参数

if $use_graphs; then
  # 如果$use_graphs 为true，使用下面的代码
  [ $nj != "`cat $srcdir/num_jobs`" ] && echo "$0: mismatch in num-jobs" && exit 1; # 检查并行线程数是否一致
  [ ! -f $srcdir/fsts.1.gz ] && echo "$0: no such file $srcdir/fsts.1.gz" && exit 1; # 检查是否存在fsts.1.gz文件

  # 模型对齐
  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
      "ark:gunzip -c $srcdir/fsts.JOB.gz|" "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
else
  # 如果$use_graphs 为false，使用下面的代码
  tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";
  # We could just use gmm-align in the next line, but it's less efficient as it compiles the
  # training graphs one by one.
  # 为了提高效率，先生成graph，然后进行对齐
  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/final.mdl  $lang/L.fst "$tra" ark:- \| \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" ark:- \
      "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir # 输出对齐结果分析结果

echo "$0: done aligning data."
