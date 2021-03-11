#!/usr/bin/env bash

# in SHARC run: source /home/ac1mmx/load_module/kaldiModules/loadModule.sh

# this recipe run G_2 model users who have test set > 0 files
# so user list is userList_0

# This recipe run a kaldi model over IDEA database.
# It takes all users but it could be set to process
# for just a subset of users (e.g. people with a 
# specific disease)
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh 

# controll if utils and steps folders exist
for dir in steps utils; do 
	[ ! -h $dir ] && ln -s $KALDI_ROOT/egs/wsj/s5/$dir ./
done

# Begin configuration section.
nj=13 # number of parallel jobs - 1 is perfect for such a small data set
lm_order=1 # language model order (n-gram quantity) - 1 is enough for digits grammar
decode_nj=13  # 20
thread_nj=1  # 4

[[ $# -ge 1 ]] && { echo "Wrong arguments!"; exit 1; }

# Removing previously created data (from last run.sh execution)
rm -rf exp mfcc data

gmm=1
dnn=1

dysIta_src=/mnt/databases/IDEA
home_dir=`pwd`

# path to local, data, feature and exp
data_dir=$home_dir/data
feat_dir=$home_dir/mfcc
exp_dir=$home_dir/exp
local_dir=$home_dir/local
dict_dir=$data_dir/dict
lang=$data_dir/lang
trset=train
etset=test

# parameters
boost_sil=1.25
scoring_opts="--word-ins-penalty 0.0"
cmvn_opts="--norm-means=false --norm-vars=false"  	# set both false if online mode
numLeavesTri1=1000
numGaussTri1=10000
numLeavesMLLT=1000
numGaussMLLT=10000
numLeavesSAT=1000
numGaussSAT=15000
nndepth=7
rbm_lrate=0.1
rbm_iter=3  	# smaller datasets should have more iterations!
hid_dim=2048	# according to the total pdfs (gmm-info tri3/final.mdl)
learn_rate=0.002
acwt=0.1 	# only affects pruning (scoring is on lattices)


echo
echo "===== PREPARING DATA ====="
echo

# run python script
$local_dir/prepare_idea_data_lang.py --database $dysIta_src --speakers_list "ALL" || exit 1
# if in SHARC server copy files from local dir
#cp -rfv  $local_dir/data $data_dir
for ss in train test; do
	ddir=$data_dir/${ss}
	utils/utt2spk_to_spk2utt.pl $ddir/utt2spk > $ddir/spk2utt
	utils/data/get_utt2dur.sh --nj $nj $ddir || exit 1
	utils/validate_data_dir.sh --no-feats $ddir || exit 1
done
# copy lexicon special for 101
#cp local/lexicon_detailed.txt $dict_dir/lexicon.txt

echo
echo "===== PREPARING LANG ====="
echo
# Prepare wordlists, etc.
utils/prepare_lang.sh $dict_dir "<SPOKEN_NOISE>" $data_dir/lang_tmp $lang || exit 1;

echo "===== MAKING lm.arpa ====="
echo

loc=`which ngram-count`;
if [ -z $loc ]; then
	if uname -a | grep 64 >/dev/null; then
		sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64
	else
		sdir=$KALDI_ROOT/tools/srilm/bin/i686
	fi
	if [ -f $sdir/ngram-count ]; then
		echo "Using SRILM language modelling tool from $sdir"
		export PATH=$PATH:$sdir
	else
		echo "SRILM toolkit is probably not installed. Instructions: tools/install_srilm.sh"
		exit 1
	fi
fi

ngram-count -order $lm_order -write-vocab $data_dir/lang_tmp/vocab-full.txt -wbdiscount -text $local_dir/corpus.txt -lm $data_dir/lang_tmp/lm.arpa

echo
echo "===== MAKING G.fst ====="
echo

# uni-gram, manually create the G.txt
Gfil=$local_dir/G_model.txt
[ ! -f $Gfil ] && echo "prepare data: no such file $Gfil" && exit 1;
fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt --keep_isymbols=false \
    --keep_osymbols=false $Gfil | fstarcsort --sort_type=ilabel > $lang/G.fst || exit 1;


# unigram generated by srlim
#cat $data_dir/lang_tmp/lm.arpa | \
#		arpa2fst - | \
#		fstprint | \
#		utils/eps2disambig.pl | \
#		utils/s2eps.pl | \
#		fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt --keep_isymbols=false --keep_osymbols=false | \
#		fstrmepsilon | \
#		fstarcsort --sort_type=ilabel > $lang/G.fst

if [ $gmm -eq 1 ]; then
	echo
	echo "===== FEATURE EXTRACTION ====="
	echo
	# ================================================================================
	# feature calculation
	# mfcc
	  for x in $trset $etset; do
		if [ ! -f $data_dir/$x/cmvn.scp ]; then
		 steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" $data_dir/$x $exp_dir/make_mfcc/$x $feat_dir/$x || exit 1
		 steps/compute_cmvn_stats.sh $data_dir/$x $exp_dir/make_mfcc/$x $feat_dir/$x || exit 1
		fi
		utils/fix_data_dir.sh $data_dir/$x || exit 1
	  done

	echo
	echo "===== GMM-HMM TRAINING ====="
	echo
	# ================================================================================
	# GMM-HMM training
	if [ ! -f $exp_dir/$trset/tri2/final.mdl ]; then
	  # Starting basic training on MFCC features for control data based on GMM/HMM
	  steps/train_mono.sh --nj $nj --cmd "$train_cmd" --cmvn-opts "$cmvn_opts" --boost-silence $boost_sil \
				$data_dir/$trset $lang $exp_dir/$trset/mono
	  steps/align_si.sh --nj $nj --cmd "$train_cmd" --boost-silence $boost_sil \
				$data_dir/$trset $lang $exp_dir/$trset/mono $exp_dir/$trset/mono_ali
	  steps/train_deltas.sh --cmd "$train_cmd" --cmvn-opts "$cmvn_opts" --boost-silence $boost_sil \
				$numLeavesTri1 $numGaussTri1 $data_dir/$trset $lang $exp_dir/$trset/mono_ali $exp_dir/$trset/tri1
	  steps/align_si.sh --nj $nj --cmd "$train_cmd" --boost-silence $boost_sil \
				$data_dir/$trset $lang $exp_dir/$trset/tri1 $exp_dir/$trset/tri1_ali
	  steps/train_lda_mllt.sh --cmd "$train_cmd" --cmvn-opts "$cmvn_opts" --boost-silence $boost_sil \
				$numLeavesMLLT $numGaussMLLT $data_dir/$trset $lang $exp_dir/$trset/tri1_ali $exp_dir/$trset/tri2
	 fi
	 
	 # SAT training
	  steps/align_si.sh --nj $nj --cmd "$train_cmd" --boost-silence $boost_sil \
				$data_dir/$trset $lang $exp_dir/$trset/tri2 $exp_dir/$trset/tri2_ali
	  steps/train_sat.sh --cmd "$train_cmd" --boost-silence $boost_sil \
				$numLeavesSAT $numGaussSAT $data_dir/$trset $lang $exp_dir/$trset/tri2_ali $exp_dir/$trset/tri3

		# decode 
	utils/mkgraph.sh $lang $exp_dir/$trset/tri2 $exp_dir/$trset/tri2/graph
	for dset in $etset; do
	  if [ ! -f $exp_dir/$trset/tri2/decode_${dset}/scoring_kaldi/best_wer ]; then
	   steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads $thread_nj --scoring-opts "$scoring_opts" --stage 0 \
			$exp_dir/$trset/tri2/graph $data_dir/${dset} $exp_dir/$trset/tri2/decode_${dset}
	  fi
	done
	# decode + SAT
	utils/mkgraph.sh $lang $exp_dir/$trset/tri3 $exp_dir/$trset/tri3/graph
	for dset in $etset; do
	  if [ ! -f $exp_dir/$trset/tri3/decode_${dset}/scoring_kaldi/best_wer ]; then
	   steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads $thread_nj --scoring-opts "$scoring_opts" --stage 0 \
			  $exp_dir/$trset/tri3/graph $data_dir/${dset} $exp_dir/$trset/tri3/decode_${dset}
	  fi
	done
	 
	grep WER $exp_dir/$trset/tri*/decode_*/scoring_kaldi/best_wer > results_gmm.txt
fi 

# ================================================================================
if [ $dnn -eq 1 ]; then
	echo
	echo "===== DNN TRAINING ====="
	echo

	echo
	echo "===== FBANK FEATURE EXTRACTION ====="
	echo
	# DNN training with FBANK features
	subfix="_fbank"
	gmmdir=$exp_dir/$trset/tri3
	dnndir=$exp_dir/$trset/dnn${subfix}

	# calculate fbank based on conf/fbank.conf
	for x in $trset $etset; do
	  utils/copy_data_dir.sh $data_dir/$x $data_dir/${x}${subfix}
	  steps/make_fbank.sh --nj $nj --cmd "$train_cmd" --fbank-config conf/fbank.conf \
			$data_dir/${x}${subfix} $exp_dir/make_mfcc/${x}${subfix} $feat_dir/${x}${subfix}
	  steps/compute_cmvn_stats.sh $data_dir/${x}${subfix} $exp_dir/make_mfcc/${x}${subfix} $feat_dir/${x}${subfix}
	  utils/fix_data_dir.sh $data_dir/${x}${subfix}
	done

	# split the training data : 90% train 10% cross-validation (held-out)
	utils/subset_data_dir_tr_cv.sh $data_dir/${trset}${subfix} $data_dir/${trset}${subfix}/tr90 $data_dir/${trset}${subfix}/cv10 || exit 1
	
	# Pre-train DBN, i.e. a stack of RBMs
	echo
	echo "===== PRE-TRAIN DBN ====="
	echo
	dir=$dnndir/pretrain
	if [ ! -f $dir/${nndepth}.dbn ]; then
		(tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
		$cuda_cmd $dir/log/pretrain_dbn.log \
				steps/nnet/pretrain_dbn.sh --nn-depth $nndepth --hid-dim $hid_dim --rbm-lrate $rbm_lrate --rbm-iter $rbm_iter --cmvn-opts "$cmvn_opts" \
				$data_dir/${trset}${subfix} $dir || exit 1;
	fi
		
	# Train the DNN optimizing per-frame cross-entropy.
	echo
	echo "===== TRAIN DNN ====="
	echo
	dir=$dnndir
	ali=${gmmdir}_ali
	if [ ! -f $ali/ali.1.gz ]; then
		steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" --boost-silence $boost_sil \
			$data_dir/${trset} $lang $gmmdir $ali || exit 1
	fi

	feature_transform=$dir/pretrain/final.feature_transform
	dbn=$dir/pretrain/${nndepth}.dbn
	(tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
	# Train
	$cuda_cmd $dir/log/train_nnet.log \
		steps/nnet/train.sh --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate $learn_rate \
		$data_dir/${trset}${subfix}/tr90 $data_dir/${trset}${subfix}/cv10 $lang $ali $ali $dir || exit 1;
		
	# Decode (reuse HCLG graph)
	for dset in $etset; do
	  if [ ! -f $dnndir/decode_${dset}/scoring_kaldi/best_wer ]; then
	   steps/nnet/decode.sh --nj $decode_nj --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
			--scoring-opts "$scoring_opts" --stage 0 \
			$gmmdir/graph $data_dir/${dset}${subfix} $dnndir/decode_${dset} || exit 1;
	  fi
	done
	grep WER $dnndir/decode_*/scoring_kaldi/best_wer > results_dnn.txt
fi
echo
	echo "===== FINISH ====="
	echo

