# idea
This repo contains the kaldi recipe for IDEA (Italian dysarthric speech) database.
For more details about IDEA and this recipe results see paper presented in SLT2021 (IDEA: AN ITALIAN DYSARTHRIC SPEECH DATABASE #1150).

It is very important to setup some files in order to make it runnable in your system.
The files to be setup are:
  - path.sh
  - run.sh
  - cmd.sh

In local folder are present some useful files:
 - analyseResultsTool.py: generates files that contain summarized results speaker by speaker.
 - G_model.txt: it contains the Grammar model for this database. The first version of IDEA
 speakers repeat single word at each time, so grammar model force language model to generates
 only one word as output. In run.sh you can decide to use a G model generated by srlim.
 - lexicon_detalied.txt: since the speakers are dysarthric, some record present some broken 
 word. e.g. the word 'casa' could be registered as 'ca silence sa' this is due to disease.
 So, this lexicon has been generated ad-hoc and present more ways to say a word. You can
 decide to use it or not in run.sh file.
 - prepare_dysita_data_lang.py: prepare all files that kaldi needs to run properly. See it
 for more details.
