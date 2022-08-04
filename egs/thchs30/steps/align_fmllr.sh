#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Computes training alignments; assumes features are (LDA+MLLT or delta+delta-delta)
# + fMLLR (probably with SAT models).
# It first computes an alignment with the final.alimdl (or the final.mdl if final.alimdl
# is not present), then does 2 iterations of fMLLR estimation.

# If you supply the --use-graphs option, it will use the training
# graphs from the source directory (where the model is).  In this
# case the number of jobs must match the source directory.


# Begin configuration section.
stage=0 # 运行特定片段开始标志
nj=4 # 并行线程数
cmd=run.pl # 运行命令方式run.pl或者queue.pl
use_graphs=false # 是否使用训练图，默认不使用
# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1" # 缩放参数
beam=10 # 搜索束的大小
retry_beam=40 # 如果搜索束失败，重新搜索束的大小
careful=false # 是否使用更加严格的搜索束
boost_silence=1.0 # factor by which to boost silence during alignment. # 在对齐时增大静音的倍数
fmllr_update_type=full # 更新类型，full或者distributed
# End configuration options.

echo "$0 $@"  # Print the command line for logging # 输出命令行，用于日志记录

[ -f path.sh ] && . ./path.sh # source the path. # 载入path.sh脚本，设置全局变量
. parse_options.sh || exit 1; # 解析-name value 类型参数

# 判断除了-name参数外，其余参数个数是否等于4，如果不等4，则报错并退出
if [ $# != 4 ]; then
   echo "usage: steps/align_fmllr.sh <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  steps/align_fmllr.sh data/train data/lang exp/tri1 exp/tri1_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --use-graphs true                                # use graphs in src-dir"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --fmllr-update-type (full|diag|offset|none)      # default full."
   exit 1;
fi

# 参数1：数据目录，参数2：语言模型目录，参数3：源目录，参数4：对齐目录
data=$1
lang=$2
srcdir=$3
dir=$4

oov=`cat $lang/oov.int` || exit 1; # 获取oov索引
silphonelist=`cat $lang/phones/silence.csl` || exit 1; # 获取静音索引
sdata=$data/split$nj # 分割数据目录

mkdir -p $dir/log # 创建日志目录
echo $nj > $dir/num_jobs # 将并行线程数写入num_jobs文件
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1; # 按照并行任务数量分割数据集

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1; # 检查语言模型和源目录的phones.txt是否一致
cp $lang/phones.txt $dir || exit 1; # 复制语言模型的phones.txt到目标目录

cp $srcdir/{tree,final.mdl} $dir || exit 1; # 复制源目录的tree和final.mdl到目标目录
cp $srcdir/final.alimdl $dir 2>/dev/null # 复制源目录的final.alimdl到目标目录，如果没有则不复制
cp $srcdir/final.occs $dir; # 复制源目录的final.occs到目标目录
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options. # 分帧选项
cp $srcdir/splice_opts $dir 2>/dev/null # frame-splicing options. # 复制分帧选项到目标目录
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null` # cmvn options. # cmvn选项
cp $srcdir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option. # 复制cmvn选项到目标目录
delta_opts=`cat $srcdir/delta_opts 2>/dev/null` # 差分选项
cp $srcdir/delta_opts $dir 2>/dev/null  # 复制差分选项到目标目录

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi # 判断是否有final.mat文件，如果有，则feat_type=lda，否则feat_type=delta
echo "$0: feature type is $feat_type" # 输出特征类型

case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";; # 如果是差分特征，则使用add-deltas添加差分特征，并将结果写入ark:-
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |" # 如果是lda特征，则使用transform-feats将特征转换为lda特征，并将结果写入ark:-
    cp $srcdir/final.mat $dir # 复制final.mat到目标目录
    cp $srcdir/full.mat $dir 2>/dev/null # 复制full.mat到目标目录，如果没有则不复制
   ;;
  *) echo "Invalid feature type $feat_type" && exit 1; # 如果特征类型不是delta或lda，则报错
esac

## Set up model and alignment model.
mdl=$srcdir/final.mdl # 设置模型文件
if [ -f $srcdir/final.alimdl ]; then
  alimdl=$srcdir/final.alimdl # 设置alimdl文件
else
  alimdl=$srcdir/final.mdl # 设置alimdl文件
fi
[ ! -f $mdl ] && echo "$0: no such model $mdl" && exit 1; # 如果模型文件不存在，则报错
alimdl_cmd="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $alimdl - |" # 设置alimdl_cmd变量，用于添加静音模型
mdl_cmd="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $mdl - |" # 设置mdl_cmd变量，用于添加静音模型


## Work out where we're getting the graphs from.
if $use_graphs; then
  [ "$nj" != "`cat $srcdir/num_jobs`" ] && \
    echo "$0: you specified --use-graphs true, but #jobs mismatch." && exit 1; # 如果使用了图，则检查nj和srcdir/num_jobs是否一致
  [ ! -f $srcdir/fsts.1.gz ] && echo "No graphs in $srcdir" && exit 1; # 如果没有图，则报错
  graphdir=$srcdir # 图目录为srcdir
else
  graphdir=$dir # 图目录为dir
  if [ $stage -le 0 ]; then
    echo "$0: compiling training graphs"
    tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|"; # 将训练文本转换为int类型，并将oov映射为unk，并写入sdata/JOB/text文件
    $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log  \
      compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/final.mdl  $lang/L.fst "$tra" \
        "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1; # 将训练图写入fsts.JOB.gz文件
  fi
fi


if [ $stage -le 1 ]; then
  echo "$0: aligning data in $data using $alimdl and speaker-independent features."
  $cmd JOB=1:$nj $dir/log/align_pass1.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$alimdl_cmd" \
    "ark:gunzip -c $graphdir/fsts.JOB.gz|" "$sifeats" "ark:|gzip -c >$dir/pre_ali.JOB.gz" || exit 1; # 对音频文件进行第一次预测，并将结果写入pre_ali.JOB.gz文件
fi

if [ $stage -le 2 ]; then
  echo "$0: computing fMLLR transforms"
  if [ "$alimdl" != "$mdl" ]; then
    $cmd JOB=1:$nj $dir/log/fmllr.JOB.log \
      ali-to-post "ark:gunzip -c $dir/pre_ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alimdl ark:- ark:- \| \
      gmm-post-to-gpost $alimdl "$sifeats" ark:- ark:- \| \
      gmm-est-fmllr-gpost --fmllr-update-type=$fmllr_update_type \
      --spk2utt=ark:$sdata/JOB/spk2utt $mdl "$sifeats" \
      ark,s,cs:- ark:$dir/trans.JOB || exit 1; 
  else
    $cmd JOB=1:$nj $dir/log/fmllr.JOB.log \
      ali-to-post "ark:gunzip -c $dir/pre_ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alimdl ark:- ark:- \| \
      gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
      --spk2utt=ark:$sdata/JOB/spk2utt $mdl "$sifeats" \
      ark,s,cs:- ark:$dir/trans.JOB || exit 1; 
  fi
fi

feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$dir/trans.JOB ark:- ark:- |" # 特征变换

if [ $stage -le 3 ]; then
  echo "$0: doing final alignment."
  $cmd JOB=1:$nj $dir/log/align_pass2.JOB.log \
    gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl_cmd" \
    "ark:gunzip -c $graphdir/fsts.JOB.gz|" "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1; # 对音频文件进行第二次对齐，并将结果写入ali.JOB.gz文件
fi

rm $dir/pre_ali.*.gz # 删除pre_ali.*.gz文件

echo "$0: done aligning data."

steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir # 运行steps/diagnostic/analyze_alignments.sh脚本，检查对齐结果是否正确

utils/summarize_warnings.pl $dir/log # 检查是否有警告信息

exit 0;
