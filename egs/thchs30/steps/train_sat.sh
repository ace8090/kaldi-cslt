#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.


# This does Speaker Adapted Training (SAT), i.e. train on
# fMLLR-adapted features.  It can be done on top of either LDA+MLLT, or
# delta and delta-delta features.  If there are no transforms supplied
# in the alignment directory, it will estimate transforms itself before
# building the tree (and in any case, it estimates transforms a number
# of times during training).


# Begin configuration section.
stage=-5 # 运行特定片段开始标志
exit_stage=-100 # you can use this to require it to exit at the
                # beginning of a specific stage.  Not all values are
                # supported. # 运行特定片段结束标记
fmllr_update_type=full # 更新类型，full或者difference
cmd=run.pl # 运行命令方式run.pl或者queue.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1" # 缩放参数
beam=10 # 搜索算法的beam大小
retry_beam=40 # 如果搜索算法没有找到最优解，则重新搜索，使用beam大小为retry_beam
careful=false # 是否使用careful搜索算法，默认为false
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment # 在对齐中增大静音概率的因子
context_opts=  # e.g. set this to "--context-width 5 --central-position 2" for quinphone. # 设置上下文的参数，例如quinphone的话，设置为"--context-width 5 --central-position 2"
realign_iters="10 20 30"; # 对齐迭代次数，例如10 20 30
fmllr_iters="2 4 6 12"; # fMLLR迭代次数，例如2 4 6 12
silence_weight=0.0 # Weight on silence in fMLLR estimation. # fMLLR估计中静音的权重
num_iters=35   # Number of iterations of training # 训练迭代次数
max_iter_inc=25 # Last iter to increase #Gauss on. # 最后一次增加高斯的迭代次数
power=0.2 # Exponent for number of gaussians according to occurrence counts # 高斯数量的指数
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves # 构建树的最终底层聚类，用于控制叶子聚类
phone_map= # 自定义的音素映射文件，例如：data/lang/phones.txt.map
train_tree=true # 是否训练树，默认为true
tree_stats_opts= # options for tree-stats, used to get tree stats # 树统计的参数，用于获取树统计信息
cluster_phones_opts= # options for cluster-phones, used to cluster phones. # 聚类音素的参数，用于聚类音素
compile_questions_opts= # options for compile-questions, used to compile questions. # 编译问题集的参数，用于编译问题
# End configuration section.

echo "$0 $@"  # Print the command line for logging # 输出命令行，用于日志记录

[ -f path.sh ] && . ./path.sh # 载入path.sh脚本，设置全局变量
. parse_options.sh || exit 1; # 解析-name value 类型参数

#  判断除了-name参数外，其余参数个数是否等于6，如果不等于6，则报错并退出
if [ $# != 6 ]; then
  echo "Usage: steps/train_sat.sh <#leaves> <#gauss> <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_sat.sh 2500 15000 data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri3b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

# 参数1：叶子节点数量，参数2：高斯数量，参数3：训练数据目录，参数4：语言模型目录，参数5：对齐目录，参数6：训练结果目录
numleaves=$1
totgauss=$2
data=$3
lang=$4
alidir=$5
dir=$6

# 判断训练数据目录是否存在，如果不存在，则报错并退出
for f in $data/feats.scp $lang/phones.txt $alidir/final.mdl $alidir/ali.1.gz; do
  [ ! -f $f ] && echo "train_sat.sh: no such file $f" && exit 1;
done

numgauss=$numleaves # 初始高斯数量为叶子节点数量
incgauss=$[($totgauss-$numgauss)/$max_iter_inc]  # per-iter #gauss increment # 每次增加高斯数量
oov=`cat $lang/oov.int` # 用于训练的OOV音素索引
nj=`cat $alidir/num_jobs` || exit 1; # 并行的任务数量
silphonelist=`cat $lang/phones/silence.csl` # 静音音素列表
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1; # 上下文无关音素列表
sdata=$data/split$nj; # 分割数据目录
splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options. # 分帧参数
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null` # cmvn参数
delta_opts=`cat $alidir/delta_opts 2>/dev/null` # delta参数
phone_map_opt= # 用于训练的音素映射文件，例如：data/lang/phones.txt.map
[ ! -z "$phone_map" ] && phone_map_opt="--phone-map='$phone_map'" # 设置音素映射文件

mkdir -p $dir/log # 创建日志目录
cp $alidir/splice_opts $dir 2>/dev/null # frame-splicing options. # 复制分帧参数到训练结果目录
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option. # 复制cmvn参数到训练结果目录
cp $alidir/delta_opts $dir 2>/dev/null # delta option. # 复制delta参数到训练结果目录

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1; # 检查语言模型目录和对齐模型目录的phones.txt文件是否一致
cp $lang/phones.txt $dir || exit 1; # 复制语言模型目录的phones.txt文件到训练结果目录

echo $nj >$dir/num_jobs # 将并行任务数量写入训练结果目录
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1; # 按照并行任务数量分割数据集

# Set up features.

if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi # 判断是否有LDA特征，如果有，则feat_type=lda，否则feat_type=delta
echo "$0: feature type is $feat_type" # 输出特征类型

## Set up speaker-independent features.
case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";; # 如果是delta特征，则将特征加上delta参数，并将特征转换为ark格式
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |" # 如果是LDA特征，则将特征加上splice参数，并将特征转换为ark格式
    cp $alidir/final.mat $dir # 复制LDA特征矩阵到训练结果目录
    cp $alidir/full.mat $dir 2>/dev/null # 复制LDA特征矩阵到训练结果目录
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1; # 如果是其他特征，则报错
esac

## Get initial fMLLR transforms (possibly from alignment dir)
# 如果有对齐模型，则从对齐模型目录中获取初始的fMLLR转换矩阵
if [ -f $alidir/trans.1 ]; then
  echo "$0: Using transforms from $alidir" 
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |" 
  cur_trans_dir=$alidir # 当前变换文件夹
else
  if [ $stage -le -5 ]; then
    echo "$0: obtaining initial fMLLR transforms since not present in $alidir" # 
    # The next line is necessary because of $silphonelist otherwise being incorrect; would require
    # old $lang dir which would require another option.  Not needed anyway. 
    [ ! -z "$phone_map" ] && \
       echo "$0: error: you must provide transforms if you use the --phone-map option." && exit 1; # 如果不存在phone_map参数，则报错
    # 
    $cmd JOB=1:$nj $dir/log/fmllr.0.JOB.log \
      ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
      weight-silence-post $silence_weight $silphonelist $alidir/final.mdl ark:- ark:- \| \
      gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
      --spk2utt=ark:$sdata/JOB/spk2utt $alidir/final.mdl "$sifeats" \
      ark:- ark:$dir/trans.JOB || exit 1; # 
  fi
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$dir/trans.JOB ark:- ark:- |" # 读取特征
  cur_trans_dir=$dir
fi

if [ $stage -le -4 ] && $train_tree; then
  # Get tree stats.
  echo "$0: Accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    acc-tree-stats $context_opts $tree_stats_opts $phone_map_opt --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
    "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1; # 计算树统计信息
  [ "`ls $dir/*.treeacc | wc -w`" -ne "$nj" ] && echo "$0: Wrong #tree-accs" && exit 1; # 判断树统计信息是否正确
  $cmd $dir/log/sum_tree_acc.log \
    sum-tree-stats $dir/treeacc $dir/*.treeacc || exit 1; # 对树统计信息求和
  rm $dir/*.treeacc # 删除树统计信息
fi

if [ $stage -le -3 ] && $train_tree; then
  echo "$0: Getting questions for tree clustering."
  # preparing questions, roots file...
  cluster-phones $cluster_phones_opts $context_opts $dir/treeacc $lang/phones/sets.int $dir/questions.int 2>$dir/log/questions.log || exit 1; # 计算树统计信息中的问题集
  cat $lang/phones/extra_questions.int >> $dir/questions.int # 将额外的问题集添加到问题集中
  compile-questions $context_opts $compile_questions_opts $lang/topo $dir/questions.int $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1; # 编译问题集

  echo "$0: Building the tree" 
  $cmd $dir/log/build_tree.log \
    build-tree $context_opts --verbose=1 --max-leaves=$numleaves \
    --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree || exit 1; # 构建树
fi

if [ $stage -le -2 ]; then
  echo "$0: Initializing the model"
  if $train_tree; then
    gmm-init-model  --write-occs=$dir/1.occs  \
      $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 2> $dir/log/init_model.log || exit 1; # 初始化模型
    grep 'no stats' $dir/log/init_model.log && echo "This is a bad warning."; # 判断是否有错误信息
    rm $dir/treeacc # 删除树统计信息
  else
    cp $alidir/tree $dir/ || exit 1; # 复制树统计信息
    $cmd JOB=1 $dir/log/init_model.log \
      gmm-init-model-flat $dir/tree $lang/topo $dir/1.mdl \
        "$feats subset-feats ark:- ark:-|" || exit 1; # 初始化模型
  fi
fi

if [ $stage -le -1 ]; then
  # Convert the alignments.
  echo "$0: Converting alignments from $alidir to use current tree"
  $cmd JOB=1:$nj $dir/log/convert.JOB.log \
    convert-ali $phone_map_opt $alidir/final.mdl $dir/1.mdl $dir/tree \
     "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1; # 转换模型
fi

[ "$exit_stage" -eq 0 ] && echo "$0: Exiting early: --exit-stage $exit_stage" && exit 0; # 如果exit_stage为0，则退出

if [ $stage -le 0 ] && [ "$realign_iters" != "" ]; then
  echo "$0: Compiling graphs of transcripts"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/1.mdl  $lang/L.fst  \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text |" \
      "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1; # 编译转移矢量
fi

x=1 # 初始化x为1
while [ $x -lt $num_iters ]; do
   echo Pass $x
  if echo $realign_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
    echo Aligning data
    mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |" # 增加静音的模型
    $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
      gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
      "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
      "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1; # 计算语音对齐
  fi

  if echo $fmllr_iters | grep -w $x >/dev/null; then
    if [ $stage -le $x ]; then
      echo Estimating fMLLR transforms
      # We estimate a transform that's additional to the previous transform;
      # we'll compose them.
      # 计算附加的变换矩阵
      $cmd JOB=1:$nj $dir/log/fmllr.$x.JOB.log \
        ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
        weight-silence-post $silence_weight $silphonelist $dir/$x.mdl ark:- ark:- \| \
        gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
        --spk2utt=ark:$sdata/JOB/spk2utt $dir/$x.mdl \
        "$feats" ark:- ark:$dir/tmp_trans.JOB || exit 1; 
      for n in `seq $nj`; do
        ! ( compose-transforms --b-is-affine=true \
          ark:$dir/tmp_trans.$n ark:$cur_trans_dir/trans.$n ark:$dir/composed_trans.$n \
          && mv $dir/composed_trans.$n $dir/trans.$n && \
          rm $dir/tmp_trans.$n ) 2>$dir/log/compose_transforms.$x.log \
          && echo "$0: Error composing transforms" && exit 1;
      done
    fi
    feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$dir/trans.JOB ark:- ark:- |" # 特征转换
    cur_trans_dir=$dir 
  fi

  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali $dir/$x.mdl "$feats" \
      "ark,s,cs:gunzip -c $dir/ali.JOB.gz|" $dir/$x.JOB.acc || exit 1; # 计算统计信息
    [ `ls $dir/$x.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1; # 判断统计信息是否正确
    $cmd $dir/log/update.$x.log \
      gmm-est --power=$power --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1; # 更新模型
    rm $dir/$x.mdl $dir/$x.*.acc # 删除旧模型和统计信息
    rm $dir/$x.occs # 删除旧统计信息
  fi
  [ $x -le $max_iter_inc ] && numgauss=$[$numgauss+$incgauss]; # 增加高斯数量
  x=$[$x+1]; # 更新x
done


if [ $stage -le $x ]; then
  # Accumulate stats for "alignment model"-- this model is
  # computed with the speaker-independent features, but matches Gaussian-for-Gaussian
  # with the final speaker-adapted model.
  $cmd JOB=1:$nj $dir/log/acc_alimdl.JOB.log \
    ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
    gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$sifeats" \
    ark,s,cs:- $dir/$x.JOB.acc || exit 1; # 计算统计信息
  [ `ls $dir/$x.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1; # 判断统计信息是否正确
  # Update model.
  $cmd $dir/log/est_alimdl.log \
    gmm-est --power=$power --remove-low-count-gaussians=false $dir/$x.mdl \
    "gmm-sum-accs - $dir/$x.*.acc|" $dir/$x.alimdl  || exit 1; # 更新模型
  rm $dir/$x.*.acc # 删除统计信息
fi

rm $dir/final.{mdl,alimdl,occs} 2>/dev/null # 删除旧模型
ln -s $x.mdl $dir/final.mdl # 更新模型
ln -s $x.occs $dir/final.occs # 更新统计信息
ln -s $x.alimdl $dir/final.alimdl # 更新模型


steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir # 解析对齐信息

utils/summarize_warnings.pl $dir/log # 汇总警告信息
(
  echo "$0: Likelihood evolution:"
  for x in `seq $[$num_iters-1]`; do
    tail -n 30 $dir/log/acc.$x.*.log | awk '/Overall avg like/{l += $(NF-3)*$(NF-1); t += $(NF-1); }
        /Overall average logdet/{d += $(NF-3)*$(NF-1); t2 += $(NF-1);}
        END{ d /= t2; l /= t; printf("%s ", d+l); } '
  done
  echo
) | tee $dir/log/summary.log


steps/info/gmm_dir_info.pl $dir # 输出模型信息

echo "$0: done training SAT system in $dir"

exit 0
