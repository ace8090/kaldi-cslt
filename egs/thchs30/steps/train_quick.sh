#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.


# Train a model on top of existing features (no feature-space learning of any
# kind is done).  This script initializes the model (i.e., the GMMs) from the
# previous system's model.  That is: for each state in the current model (after
# tree building), it chooses the closes state in the old model, judging the
# similarities based on overlap of counts in the tree stats.

# Begin configuration..
cmd=run.pl # 运行命令方式run.pl或者queue.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1" # 缩放参数
realign_iters="10 15"; # Only realign twice. # 只对齐两次，在第10和15次迭代中对齐
num_iters=20    # Number of iterations of training # 训练迭代次数
maxiterinc=15 # Last iter to increase #Gauss on. # 在最后一次迭代中增加高斯数量的迭代次数
batch_size=750 # batch size to use while compiling graphs... memory/speed tradeoff. # 在编译图时使用的批量大小
beam=10 # alignment beam. # 对齐的束大小
retry_beam=40 # 如果第一次尝试失败，尝试运行这个大小的束
stage=-5 # 运行特定片段开始标志
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves # 对于最终的底层群簇，用于构建树的群簇阈值
# End configuration section.

echo "$0 $@"  # Print the command line for logging # 输出命令行，用于日志记录

[ -f path.sh ] && . ./path.sh # 载入path.sh脚本，设置全局变量
. parse_options.sh || exit 1; # 解析-name value 类型参数

# 判断除了-name参数外，其余参数个数是否等于6，如果不等于6，则报错并退出
if [ $# != 6 ]; then
  echo "Usage: steps/train_quick.sh <num-leaves> <num-gauss> <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_quick.sh 2500 15000 data/train_si284 data/lang exp/tri3c_ali_si284 exp/tri4b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

# 参数1：需要训练的高斯数量，参数2：需要训练的高斯数量，参数3：训练数据路径，参数4：语言模型路径，参数5：对齐路径，参数6：训练结果路径
numleaves=$1
totgauss=$2
data=$3
lang=$4
alidir=$5
dir=$6

for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1; # 判断文件是否存在，如果不存在，则报错并退出
done

# Set various variables.
oov=`cat $lang/oov.int` # 读取语言模型中的oov词的索引号
silphonelist=`cat $lang/phones/silence.csl` # 读取语言模型中的silence词的索引号
ciphonelist=`cat $lang/phones/context_indep.csl` # 读取语言模型中的context_indep词的索引号
numgauss=$[totgauss/2] # Start with half the total number of Gaussians.  We won't have # 初始化高斯数量，将总高斯数量的一半分配给模型
  # to mix up much probably, as we're initializing with the old (already mixed-up) pdf's.  
[ $numgauss -lt $numleaves ] && numgauss=$numleaves # 如果高斯数量小于叶子节点数量，则将高斯数量设置为叶子节点数量
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss # 每次迭代中增加的高斯数量
nj=`cat $alidir/num_jobs` || exit 1; # 读取对齐路径中的num_jobs文件，如果不存在，则报错并退出
sdata=$data/split$nj # 并行的任务数量
splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options. # 分帧选项
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null` # cmvn选项
delta_opts=`cat $alidir/delta_opts 2>/dev/null` # delta选项

mkdir -p $dir/log # 创建日志目录
echo $nj >$dir/num_jobs # 将任务数量写入num_jobs文件
cp $alidir/splice_opts $dir 2>/dev/null # 将分帧选项复制到目标目录
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option. # 将cmvn选项复制到目标目录
cp $alidir/delta_opts $dir 2>/dev/null # delta选项复制到目标目录
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1; # 按照并行任务数量分割数据集

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1; # 检查语言模型目录和对齐模型目录的phones.txt文件是否一致
cp $lang/phones.txt $dir || exit 1; # 将语言模型目录中的phones.txt文件复制到目标目录

## Set up features.
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi # 判断是否有final.mat文件，如果有，则feat_type为lda，否则为delta
echo "$0: feature type is $feat_type" # 输出特征类型

case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";; # 如果是delta特征，则将特征加上delta选项，并将特征转换为ark格式
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |" # 如果是lda特征，则将特征加上splice选项，并将特征转换为ark格式
    cp $alidir/final.mat $dir    # 将lda模型的final.mat文件复制到目标目录
    cp $alidir/full.mat $dir 2>/dev/null # 将lda模型的full.mat文件复制到目标目录
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1; # 如果特征类型不是delta或lda，则报错并退出
esac
if [ -f $alidir/trans.1 ]; then
  echo "$0: using transforms from $alidir" 
  ln.pl $alidir/trans.* $dir # Link them to dest dir. 
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$dir/trans.JOB ark:- ark:- |" # 特征转换
else
  feats="$sifeats"
fi
##


if [ $stage -le -5 ]; then
  echo "$0: accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    acc-tree-stats  --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
    "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1; # 将对齐结果转换为树统计结果
  [ "`ls $dir/*.treeacc | wc -w`" -ne "$nj" ] && echo "$0: Wrong #tree-stats" && exit 1; # 判断树统计结果文件数量是否等于并行任务数量
  sum-tree-stats $dir/treeacc $dir/*.treeacc 2>$dir/log/sum_tree_acc.log || exit 1; # 将树统计结果汇总到一个文件中
  rm $dir/*.treeacc # 删除树统计结果文件
fi

if [ $stage -le -4 ]; then
  echo "$0: Getting questions for tree clustering."
  # preparing questions, roots file...
  cluster-phones $dir/treeacc $lang/phones/sets.int $dir/questions.int 2> $dir/log/questions.log || exit 1; # 将树统计结果和语言模型目录中的phones/sets.int文件和$dir/questions.int文件进行聚类
  cat $lang/phones/extra_questions.int >> $dir/questions.int # 将语言模型目录中的phones/extra_questions.int文件追加到$dir/questions.int文件中
  compile-questions $lang/topo $dir/questions.int $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1; # 将$dir/questions.int文件转换为$dir/questions.qst文件

  echo "$0: Building the tree"
  $cmd $dir/log/build_tree.log \
    build-tree --verbose=1 --max-leaves=$numleaves \
    --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree || exit 1; # 将树统计结果和语言模型目录中的phones/roots.int文件和$dir/questions.qst文件和语言模型目录中的topo文件进行聚类
fi

if [ $stage -le -3 ]; then
  echo "$0: Initializing the model"

  # The gmm-init-model command (with more than the normal # of command-line args)
  # will initialize the p.d.f.'s to the p.d.f.'s in the alignment model.

  gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $lang/topo $dir/tmp.mdl $alidir/tree $alidir/final.mdl  \
    2>$dir/log/init_model.log || exit 1; # 初始化模型

  grep 'no stats' $dir/log/init_model.log && echo "$0: This is a bad warning."; # 判断是否有no stats的警告信息
  rm $dir/treeacc # 删除树统计结果文件
fi

if [ $stage -le -2 ]; then
  echo "$0: mixing up old model."
  # We do both mixing-down and mixing-up to get the target #Gauss in each state,
  # since the initial model may have either more or fewer Gaussians than we want.
  gmm-mixup --mix-down=$numgauss --mix-up=$numgauss $dir/tmp.mdl $dir/1.occs $dir/1.mdl \
    2> $dir/log/mixup.log || exit 1; # 将初始模型中的模型数量更改为$numgauss
  rm $dir/tmp.mdl  #  删除临时模型文件
fi

# Convert alignments to the new tree.
if [ $stage -le -1 ]; then
  echo "$0: converting old alignments"
  $cmd JOB=1:$nj $dir/log/convert.JOB.log \
    convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree \
    "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1; # 将原始对齐结果转换为新的对齐结果
fi

if [ $stage -le 0 ]; then
  echo "$0: compiling training graphs"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int --batch-size=$batch_size $dir/tree $dir/1.mdl $lang/L.fst  \
    "ark:sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text |" \
    "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1; # 将训练文本转换为训练图
fi

x=1 # 初始化x为1
while [ $x -lt $num_iters ]; do
  echo "$0: pass $x"
  if echo $realign_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
    echo "$0: aligning data"
    $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
      gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam $dir/$x.mdl \
      "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" \
      || exit 1; # 对齐数据
  fi
  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali  $dir/$x.mdl "$feats" \
      "ark,s,cs:gunzip -c $dir/ali.JOB.gz|"  $dir/$x.JOB.acc || exit 1; # 统计结果
    [ "`ls $dir/$x.*.acc | wc -w`" -ne "$nj" ] && echo "$0: wrong #accs" && exit 1; # 判断是否有$nj个acc文件
    $cmd $dir/log/update.$x.log \
      gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1; # 更新模型
    rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs # 删除旧模型和统计结果文件
  fi
  [[ $x -le $maxiterinc ]] && numgauss=$[$numgauss+$incgauss]; # 增加模型中的Gaussians数量
  x=$[$x+1]; # 更新x
done

if [ -f $alidir/trans.1 ]; then
  echo "$0: estimating alignment model"
  $cmd JOB=1:$nj $dir/log/acc_alimdl.JOB.log \
    ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
    gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$sifeats" \
    ark,s,cs:- $dir/$x.JOB.acc || exit 1; # 统计结果
  [ "`ls $dir/$x.*.acc | wc -w`" -ne "$nj" ] && echo "$0: wrong #accs" && exit 1; # 判断是否有$nj个acc文件

  $cmd $dir/log/est_alimdl.log \
    gmm-est --write-occs=$dir/final.occs --remove-low-count-gaussians=false $dir/$x.mdl \
    "gmm-sum-accs - $dir/$x.*.acc|" $dir/$x.alimdl || exit 1; # 更新模型
  rm $dir/$x.*.acc # 删除统计结果文件
  rm $dir/final.alimdl 2>/dev/null  # 删除旧模型文件
  ln -s $x.alimdl $dir/final.alimdl # 建立新模型文件
fi

rm $dir/final.mdl 2>/dev/null # 删除旧模型文件
ln -s $x.mdl $dir/final.mdl # 建立新模型文件

echo Done
