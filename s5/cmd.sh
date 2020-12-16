# Setting local system jobs (local CPU - no external clusters)
#export train_cmd=run.pl
#export decode_cmd=run.pl
#export cuda_cmd=run.pl
#export mkgraph_cmd=run.pl

memval=4G
#h_rt="08:00:00"
#mem="-l mem=$mem_val,h_rt=$h_rt"

export train_cmd="queue.pl --mem $memval"
export decode_cmd="queue.pl --mem $memval"
export mkgraph_cmd="queue.pl --mem $memval"
export cuda_cmd="queue.pl --mem $memval --gpu 1"
export cmd="queue.pl --mem $memval"