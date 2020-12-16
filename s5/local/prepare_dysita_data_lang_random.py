#!/usr/bin/env python3
"""
Give in input:
 - path of IDEA database

Furthermore it could be pass as parameter a other flag:
 --disease: a specific or set of disease to be analyse
 --speakers_list: speakers id list of speakers that we want analyse
 --random_train_test: random wave file selection for test and train set
 --gender: M or F or both
 --database: database path
The two flags could be use at the same time, it means that the software will find the speakers who have that ID and that
specific disease. If a ID user does not appear in that disease folder, it will be ignored and a warning message appears. 

Program able to generate a all the files needed for Kaldi recipe
The files generated are:
 - /data/{test, train}/text
 - /data/{test, train}/utt2spk
 - /data/{test, train}/spk2utt
 - /data/{test, train}/wav.scp
 - /data/{test, train}/segments
 - /data/local/corpus.txt
 - /data/local/dict/lexicon.txt
 - /data/local/dict/nonsilence_phones.txt
 - /data/local/dict/silence_phones.txt
 - /data/local/dict/optional_silence.txt

All these files are generate in this way:
    For each speaker every single word pronounced is evaluated;
    For each word we see how many records (without notes) are available;
    The records are divided in test and train sets based on 2:1 rate,
    e.g if for the word 'casa' of speaker 11 we have 3 recs, they are
    splitted in 1 test ad 2 in train; 5 recs -> 4 in train and 1 in test...
    We evaluate each speaker individually and generate a test and train sets
    that countains all the sub-test and train sets of each speaker.  
"""

import textgrid
import os
#import matplotlib.figure
#import matplotlib.patches
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#from tkinter.ttk import *
#from tkinter import *
#import numpy as np
import pandas
import math
import sys
import random
import argparse

################## main variables
ALL_DISEASES = {'ALS': 'Amyotrophic_Lateral_Sclerosis',
                'PRK': 'Parkinson_Disease',
                'HC': 'Huntington\'s_Chorea',
                'STR': 'Stroke',
                'MS': 'Multiple_Sclerosis',
                'TBI': 'Traumatic_Brain_Injury',
                'MD': 'Myotonic_Dystrophy',
                'ATX': 'Ataxia',
                'NEU': 'Neuropathy',
                'OTH': 'Other',
                'ALL': 'All diseases'}

class Word:
    def __init__(self, word, phonems):
        self.word = word
        self.phonems = phonems
    def getPhonems(self):
        return self.phonems

class Phonemes:
    def __init__(self):
        self.listOfPhonemes = {}
    def getList(self):
        return self.listOfPhonemes
    def insert(self, phoneme):
        if not phoneme in self.listOfPhonemes.keys():
            self.listOfPhonemes[phoneme] = 1
        else:
            self.listOfPhonemes[phoneme] += 1
    def isPhonemeInList(self, phoneme):
        if not phoneme in self.listOfPhonemes.keys():
            return False
        else:
            return True

class Record:
    def __init__(self, id, path, word, start, end, spk, notes, duration):
        self.path = path
        self.id = id
        self.word = word
        self.start = start
        self.end = end
        self.speaker = spk
        self.notes = notes
        self.duration = duration
        self.voice = end-start
    def haveNotes(self):
        #print(self.notes)
        return sum(self.notes)>0

class Speaker:
    def __init__(self, id, dbDir, disease, gender):
        self.words = []
        self.records = []
        self.emptyFiles = 0
        self.TestAndTrain = {}
        self.TestAndTrain['test'] = []
        self.TestAndTrain['train'] = []
        if id==-1:
            self.id = 'ALL'
        else:
            self.id = id
            self.folderPath = dbDir+'/'+disease+'/'+gender+'/'+str(id)+'/single'
            wordsFolders = os.listdir(self.folderPath)
            # take just the folder
            index = 0 
            for word in wordsFolders:
                actualPath = self.folderPath+'/'+word
                if ( not os.path.isdir(actualPath)):
                    wordsFolders.pop(index)
                else:
                    # case of folder
                    if word == 'emptyWaveFiles':
                        # count number of wav file
                        filesList = os.listdir(actualPath)
                        for file in filesList:
                            if 'wav' in file:
                                self.emptyFiles += 1
                    else:
                        # parse TextGrid file
                        filesList = os.listdir(actualPath)
                        totNumOfRec = 0
                        totSecOfRec = 0.0
                        numOfRecUsable = 0
                        secOfRecUsable = 0.0
                        notes = [0] * 9
                        files = []
                        recWord = []
                        for file in filesList:
                            if 'TextGrid' in file:
                                files += [actualPath+'/'+file.replace('TextGrid', 'wav')]
                                annotation = self.parseTextGrid(actualPath+'/'+file)
                                totNumOfRec += 1
                                totSecOfRec += annotation[0]        # duration of entire rec
                                numOfRecUsable += annotation[1]
                                secOfRecUsable += annotation[2]
                                notes = [sum(x) for x in zip(notes,annotation[3])]
                                self.records.append([actualPath+'/'+file.replace('TextGrid', 'wav'), annotation[3]])
                                recWord.append(Record(file.split('.')[0], actualPath+'/'+file.replace('TextGrid', 'wav'), word, annotation[4], annotation[5], str(id), annotation[3], annotation[0]))
                        wordTmp = [word, totNumOfRec, numOfRecUsable, totSecOfRec, secOfRecUsable, notes, files, recWord]
                        self.words = self.words+[wordTmp]
                """print(interval)
                print(interval.mark)
                print(interval.minTime)
                print(interval.maxTime)
                print(interval.bounds())
                print(interval.duration())"""
                index += 1
    def getTestAndTrainSet(self, random_train_test):
        for word in self.words:
            # how many rec has a word
            #print(word)
            recList = word[7]
            numRec = len(recList)
            # use just rec without notes
            recWONList = []
            for rec in recList:
                if not rec.haveNotes():
                    recWONList.append(rec)
            if len(recWONList)>2:
                if random_train_test:
                    # make the splitting between test and train randomly
                    tmp = recWONList.copy()
                    recWONList = []
                    for i in range(len(tmp)):
                        recWONList.append(random.choice(tmp))
                        tmp.remove(recWONList[-1])
                # split in train and test based on 2:1 rate
                for i in range(len(recWONList)):
                    if i<int(math.ceil(len(recWONList)*2/3)):
                        self.TestAndTrain['train'].append(recWONList[i])
                    else:
                        self.TestAndTrain['test'].append(recWONList[i])
            else:
                for i in range(len(recWONList)):
                    self.TestAndTrain['train'].append(recWONList[i])
        #print('Train: '+str(self.TestAndTrain['train']))
        #print('Test: '+str(self.TestAndTrain['test']))
        print('Train len: '+str(len(self.TestAndTrain['train']))+", sec of voice: "+str(self.getVoiceHoursOfTrain())+", tot sec: "+str(self.getTotHoursOfTrain()))
        print('Test len: '+str(len(self.TestAndTrain['test']))+", sec of voice: "+str(self.getVoiceHoursOfTest())+", tot sec: "+str(self.getTotHoursOfTest()))
        return self.TestAndTrain
    def getTotHoursOfTrain(self):
        ret = 0
        for r in self.TestAndTrain['train']:
            ret += r.duration
        return ret
    def getVoiceHoursOfTrain(self):
        ret = 0
        for r in self.TestAndTrain['train']:
            ret += r.voice
        return ret
    def getTotHoursOfTest(self):
        ret = 0
        for r in self.TestAndTrain['test']:
            ret += r.duration
        return ret
    def getVoiceHoursOfTest(self):
        ret = 0
        for r in self.TestAndTrain['test']:
            ret += r.voice
        return ret
    def mergeSpeakers(self, sp2):
        self.emptyFiles += sp2.emptyFiles
        self.words += sp2.words
        self.records += sp2.records
    def findRecWithNotes(self, note, opt):
        # return a list of rec wich has the note
        # indicated by note input encoded in this
        # way:
        # note = [
        #         1/0 -> truncaded
        #          1/0 -> substitution
        #         1/0 -> repetition
        #         1/0 -> corrupted
        #         1/0 -> background noise
        #         1/0 -> not usable
        #         1/0 -> word splitted
        #         1/0 -> general notes 
        #          1/0 -> without notes]
        # opt vector indicates what kind of set we are
        # finding for: union, intersection or complement
        # opt = 0 -> union
        # opt = 1 -> intersection
        # opt = 2 -> complement
        #print('Note: '+str(note))
        ret = []
        if sum(note)>1:
            # case of more classes selected
            # union
            if opt==0:
                for rec in self.records:
                    selected = False
                    for (o, r) in zip(note[:8], rec[1][0:8]):
                        if o == 1 and o == r:
                            ret.append(rec)
                            selected = True
                            #print('Rec: '+str(rec))
                            break
                    if note[8]==1 and rec[1][0:8]==[0]*8:
                        ret.append(rec)
                        selected = True
                    if not selected:
                            print('Excluded: Note:'+str(note)+', rec:'+str(rec)+'; o='+str(o)+', r='+str(r))
                        #print('Rec: '+str(rec))
            # complement
            elif opt==2:
                for rec in self.records:
                    for o in enumerate(note[:8]):
                        if sum(rec[1][0:8])==1 and o[1]==1 and o[1]==rec[1][o[0]]:
                            ret.append(rec)
                            #print('Rec: '+str(rec))
                    if note[8]==1 and rec[1][0:8]==[0]*8:
                        ret.append(rec)
                        #print('Rec: '+str(rec))
            # intersection
            elif opt==1:
                for r in self.records:
                    if r[1][0:8] == note[:8]:
                        ret.append(r)
                        #print('Rec: '+str(r))
                    if note[8]==1 and r[1][0:8]==[0]*8:
                        ret.append(r)
                        #print('Rec: '+str(r))
        else:
            # case of one class vs others
            # union
            if opt==0:
                for rec in self.records:
                    #print('Note:'+str(note)+', rec[1]:'+str(rec[1]))
                    for (o, r) in zip(note[:8], rec[1][0:8]):
                        if o == 1 and o == r:
                            ret.append(rec)
                            #print('Rec: '+str(rec))
                            break
                    if note[8]==1 and rec[1][0:8]==[0]*8:
                        ret.append(rec)
                        #print('Rec: '+str(rec))
            # complement
            elif opt==1:
                for rec in self.records:
                    if sum(rec[1][0:8])>1:
                        for (o, r) in zip(note[:8], rec[1][0:8]):
                            if o == 1 and o == r:
                                ret.append(rec)
                                #print('Rec: '+str(rec))
                                break
                    if note[8]==1 and rec[1][0:8]==[0]*8:
                        ret.append(rec)
                        #print('Rec: '+str(rec))
            # intersection
            elif opt==2:
                for r in self.records:
                    if r[1][0:8] == note[:8]:
                        ret.append(r)
                        #print('Rec: '+r)
                    if note[8]==1 and r[1][0:8]==[0]*8:
                        ret.append(r)
                        #print('Rec: '+str(r))
        return ret
    def parseTextGrid(self, pathFile):
        #create a Textgrid obbject, name it if required.
        annotation = textgrid.TextGrid('file')
        #read the texgrid file
        #print('parse file:'+ pathFile)
        annotation.read(pathFile)
        # control tiers
        # the first two elements are:
        #    - duration of entier rec
        #    - usable rec (by def is 1 because we assume that every rec are usable)
        ret = [annotation.maxTime, 1]
        listOfTiers = annotation.getNames()
        tmpVoiceSec = 0.0
        tmpStart = 0.0
        tmpEnd = 0.0
        tmpNotes = [0] * 9
        for tier in listOfTiers:
            #print("\ttier name: "+tier)
            for l in range(len(annotation.getList(tier))):
                for j, interval in enumerate(annotation.getList(tier)[l]):
                    # control mark of tier
                    if tier == 'words':
                        if interval.mark != 'nonspeech':
                            tmpVoiceSec = interval.duration()
                            tmpStart = interval.minTime
                            tmpEnd = interval.maxTime
                            #print("File: "+pathFile)
                            #print("    -> "+interval.mark+", "+str(tmpStart)+", "+str(tmpEnd))
                    else:
                        if interval.mark == 'truncated':
                            tmpNotes[0] = 1
                        if interval.mark == 'substitution':
                            tmpNotes[1] = 1
                        if interval.mark == 'repetition':
                            tmpNotes[2] = 1
                        if interval.mark == 'corrupted':
                            tmpNotes[3] = 1
                            tmpVoiceSec = 0.0
                            #ret[1] = 0
                        if interval.mark == 'background noise':
                            tmpNotes[4] = 1
                        if interval.mark == 'not usable':
                            tmpNotes[5] = 1
                            tmpVoiceSec = 0.0
                            ret[1] = 0
                        if interval.mark == 'word splitted':
                            tmpNotes[6] = 1
                        if interval.mark == 'general notes':
                            tmpNotes[7] = 1
        # tmpNotes[8] means that the rec has at least one note different to 'not usable'
        if (sum(tmpNotes)-tmpNotes[5])>0:
            tmpNotes[8] = 1
        ret += [tmpVoiceSec]
        ret += [tmpNotes]
        ret += [tmpStart]
        ret += [tmpEnd]
        return ret
    def getAllRec(self):
        ret = 0
        for w in self.words:
            ret += w[1]
        return ret + self.emptyFiles
    def getAllRecUsable(self):
        ret = 0
        for w in self.words:
            ret += w[2]
        return ret
    def getSecOfAllRec(self):
        ret = 0.0
        for w in self.words:
            ret += w[3]
        return ret
    def getSecOfAllRecUsable(self):
        ret = 0.0
        for w in self.words:
            ret += w[4]
        return ret
    def getAllRecNotUsable(self):
        ret = 0
        for w in self.words:
            ret += w[1] - w[2]
        return ret
    def getAllRecUsableWithoutNotes(self):
        ret = 0
        for w in self.words:
            ret += w[2] - (w[5])[8]
            #ret += w[2] - sum(w[5]) + (w[5])[5]
            if (w[2] - (w[5])[8]) < 0 :
                print("Error: control word "+w[0]+" of speaker "+self.id+"; Number of usable with or without notes not coincides")
                print(w)
        return ret
    def getAllRecUsableWithNotes(self):
        ret = 0
        for w in self.words:
            ret += (w[5])[8]
            #ret += sum(w[5])-(w[5])[5]
        return ret
    def getAllRecNotes(self, note):
        if note == 'truncated':
            index = 0
        elif note == 'substitution':
            index = 1
        elif note == 'repetition':
            index = 2
        elif note == 'corrupted':
            index = 3
        elif note == 'background noise':
            index = 4
        elif note == 'not usable':
            index = 5
        elif note == 'word splitted':
            index = 6
        elif note == 'general notes':
            index = 7
        else:
            return -1
        ret = 0
        for w in self.words:
            note = w[5]
            ret += note[index]
        return ret
    def getRecDataForPie(self):
        ret = []
        ret += [self.emptyFiles]
        ret += [self.getAllRecNotUsable()]
        ret += [self.getAllRecUsableWithoutNotes()]
        ret += [self.getAllRecUsableWithNotes()]
        return ret
    def getRecWithNotesDataForPie(self):
        ret = []
        notesL = ['truncated', 'substitution', 'repetition', 'corrupted', 'background noise', 'word splitted', 'general notes']
        for n in notesL:
            ret += [self.getAllRecNotes(n)]
        return ret
    def getSecDataForPie(self):
        ret = []
        ret += [self.getSecOfAllRecUsable()]
        ret += [self.getSecOfAllRec()-self.getSecOfAllRecUsable()]
        return ret
    def printAllWordsOccurences(self):
        for w in self.words:
            print(w[0]+': '+str(w[1]))

def genRecID(rec):
    # generate utt_id given a rec object
    # utt_id is composed by:
    #  - spk id
    #  - filename
    return rec.speaker+'_'+rec.id

def genUttID(rec):
    # generate utt_id given a rec object
    # utt_id is composed by:
    #  - spk id
    #  - filename
    #  - start sec
    #  - end sec
    return genRecID(rec)+'_'+str("%3.i" % (rec.start*100)).replace(' ', '0')+'_'+str("%3.i" % (rec.end*100)).replace(' ', '0')

def generateTextFile(path, listRec):
    # generate text file using list of rec objects
    if os.path.isdir(path):
        textFile = path+'/'+'text'
        with open(textFile, 'w') as tf:
            for rec in sorted(listRec, key=genRecID):
                uttId = genUttID(rec)
                tf.write(uttId+' '+rec.word+'\n')

def generateUtt2spkFile(path, listRec):
    # generate utt2spk file using list of rec objects
    if os.path.isdir(path):
        textFile = path+'/'+'utt2spk'
        with open(textFile, 'w') as tf:
            for rec in sorted(listRec, key=genRecID):
                uttId = genUttID(rec)
                tf.write(uttId+' '+rec.speaker+'\n')

def generateSpk2uttFile(path, listRec):
    # generate spk2utt file using list of rec objects
    if os.path.isdir(path):
        textFile = path+'/'+'spk2utt'
        with open(textFile, 'w') as tf:
            spIdTmp = ''
            firstTime = True
            for rec in sorted(listRec, key=genRecID):
                uttId = genUttID(rec)
                if rec.speaker==spIdTmp:
                        tf.write(' '+uttId)
                else:
                    if not firstTime:
                        tf.write('\n')
                    firstTime = False
                    tf.write(rec.speaker+' '+uttId)
                    spIdTmp = rec.speaker

def generateWavscpFile(path, listRec):
    # generate wav.scp file using list of rec objects
    if os.path.isdir(path):
        textFile = path+'/'+'wav.scp'
        with open(textFile, 'w') as tf:
            for rec in sorted(listRec, key=genRecID):
                recId = genRecID(rec)
                tf.write(recId+' '+rec.path+'\n')

def generateSegmentsFile(path, listRec):
    # generate segments file using list of rec objects
    if os.path.isdir(path):
        segm = path+'/'+'segments'
        with open(segm, 'w') as tf:
            for rec in sorted(listRec, key=genRecID):
                uttId = genUttID(rec)
                recId = genRecID(rec)
                tf.write(uttId+' '+recId+' '+str(rec.start)+' '+str(rec.end)+'\n')

def generateCorpusFile(path, listWords):
    # generate text file using list of rec objects
    if os.path.isdir(path):
        textFile = path+'/'+'corpus.txt'
        with open(textFile, 'w') as tf:
            for w in listWords:
                tf.write(w+'\n')

def generateLexiconFile(path, listWords):
    # generate text file using list of rec objects
    if os.path.isdir(path):
        textFile = path+'/'+'lexicon.txt'
        with open(textFile, 'w') as tf:
            tf.write('<SPOKEN_NOISE> sil\n')
            for w in listWords:
                phons = ''
                for f in listWords[w]:
                    phons += f+' '
                tf.write(w+' '+phons+'\n')

def controlIfAllWordsArePronounced(listRec, listWord):
    # see if listWord and listRc have some miss matching
    wordsNounPronounced = []
    wordsMissedInDict = []
    wordsPronounced = []
    for sets in listRec:
        for rec in listRec[sets]:
            if not rec.word in listWord:
                if not rec.word in wordsMissedInDict:
                    wordsMissedInDict.append(rec.word)
            else:
                wordsPronounced.append(rec.word)
    #if len(wordsPronounced)!=len(listWord):
        # control which words are not pronounced
        #for w in listWord:
            #if not w in wordsPronounced:
            #    print('Word not pronounced: '+w)
    if len(wordsMissedInDict)>0:
        for w in wordsMissedInDict:
            print('Word missed in dict: '+w)
    ret = {}
    for w in wordsPronounced:
        ret[w] = listWord[w]
    return ret

def generateSilenceAndOptimalFiles(path):
    files = [path+'/silence_phones.txt', path+'/optional_silence.txt']
    for out in files:
        with open(out, 'w') as f:
            f.write('sil\n')

def generateNonsilenceFile(path, wordList):
    #print("List of words: "+str(list(wordList)))
    phonemesList = Phonemes()
    for w in wordList.items():
        #print("word: "+str(w))
        for p in w[1]:
            #print("phone insert: "+str(p))
            phonemesList.insert(p)
    #print("Phonems: "+str(phonemesList.listOfPhonemes.keys()))
    with open(path+'/nonsilence_phones.txt', 'w') as f:
        for p in list(phonemesList.listOfPhonemes.keys()):
            f.write(p+'\n')


def generateKaldiFile(testAndTrainDict, folderName, wordsList, phonemesList):
    # generate Kaldi files and folders based on test and train dictionary
    # generate out folder
    mainPath = os.getcwd()
    folderPath = mainPath#+'/'+folderName
    #if os.path.isdir(folderPath):
        # folder already exist
    #    print('Folder already exist, please give another name')
    #    print(folderPath)
    #    return
    # create folders
    pathData = folderPath+'/'+'data'
    pathDataLocal = folderPath+'/'+'local'
    pathDataTrain = pathData+'/'+'train'
    pathDataTest = pathData+'/'+'test'
    pathDataLocalDict = pathData+'/'+'dict'
    #pathAudiowave = folderPath+'/'+'audio_wave'
    #pathAudiowaveTrain = pathAudiowave+'/'+'train'
    #pathAudiowaveTest = pathAudiowave+'/'+'test'
    #os.mkdir(folderPath)
    # creation of folders
    if not os.path.isdir(pathData):
        os.mkdir(pathData)
    #os.mkdir(pathAudiowave)
    if not os.path.isdir(pathDataTrain):
        os.mkdir(pathDataTrain)
    if not os.path.isdir(pathDataTest):
        os.mkdir(pathDataTest)
    if not os.path.isdir(pathDataLocal):
        os.mkdir(pathDataLocal)
    if not os.path.isdir(pathDataLocalDict):
        os.mkdir(pathDataLocalDict)
    #os.mkdir(pathAudiowaveTrain)
    #os.mkdir(pathAudiowaveTest)

    # **** generate acoustic data **** #
    for sets in testAndTrainDict.items():
        # generate text
        generateTextFile(pathData+'/'+sets[0], sets[1])
        # generate utt2spk
        generateUtt2spkFile(pathData+'/'+sets[0], sets[1])
        # generate spk2utt
        #generateSpk2uttFile(pathData+'/'+sets[0], sets[1])
        # generate wav.scp
        generateWavscpFile(pathData+'/'+sets[0], sets[1])
        # generate segments
        generateSegmentsFile(pathData+'/'+sets[0], sets[1])
    # **** generate languade data **** #
    # annotation of words used by speaker
    wordsPronounced = controlIfAllWordsArePronounced(testAndTrainDict, wordsList)
    # generate corpus
    generateCorpusFile(pathDataLocal, wordsPronounced)

    # generate lexicon
    generateLexiconFile(pathDataLocalDict, wordsPronounced)

    # generate silence_phone and optimal_silence
    generateSilenceAndOptimalFiles(pathDataLocalDict)

    # generate nonsilence_phone
    generateNonsilenceFile(pathDataLocalDict, wordsPronounced)

def errorMesg():
    print("Error! The program has wrong argument")

def parseSpeaker(listOfSpeaker, spIdCode, dbDir, disease, gend):
    spkDir = dbDir+'/'+disease+'/'+gend+'/'+spIdCode
    if os.path.exists(spkDir):
        if os.path.isdir(spkDir):
            #listOfSpeaker = {}
            #speakerKeysValues = []
            #for usId in os.listdir(os.getcwd()):
            #    if ( os.path.isdir(os.getcwd()+'/'+usId)):
            listOfSpeaker[spIdCode] = Speaker(spIdCode, dbDir, disease, gend)
            #speakerKeysValues += [str(usId)]
            sp1 = listOfSpeaker[str(spIdCode)]
            print('Speaker id: '+str(sp1.id))
            print('Gender: ', gend)
            print('Disease: ', disease)
            print('\tAll rec: '+str(sp1.getAllRec()))
            print('\tEmpty files: '+str(sp1.emptyFiles))
            print('\tNum of rec NOT usable: '+str(sp1.getAllRecNotUsable()))
            #print('\tAll sec: '+str(sp1.getSecOfAllRec()))
            print('\tNum of rec usable: '+str(sp1.getAllRecUsable()))
            print('\tNum of rec usable whitOUT notes: '+str(sp1.getAllRecUsableWithoutNotes()))
            print('\tNum of rec usable whit notes: '+str(sp1.getAllRecUsableWithNotes()))
            #print('\tSec of voice: '+str(sp1.getSecOfAllRecUsable()))
            notesL = ['truncated', 'substitution', 'repetition', 'corrupted', 'background noise', 'not usable', 'word splitted', 'general notes']
            for n in notesL:
                print('\tNum rec with '+n+': '+str(sp1.getAllRecNotes(n)))
    else:
        print("Error! Speaker ID not correct: "+spIdCode)
        print('The path does not exist: '+spkDir)
        exit()
        
def parse_random_train_test(args):
    #random_train_test = args.random_train_test.replace(" ", "")
    #print('random: ', random)
    ret = False
    if args.random_train_test == 1:
        ret = True
    elif args.random_train_test == 0:
        ret = False
    else:
        print("""Warning! The random_train_test flag should be 0 or 1.
        Otherwise it will be set with default value: 0""")
        ret = False
    return ret

def parse_database(args):
    if len(args.database) == 0:
        return ''
    database = args.database.replace(" ", "")
    return database
    
def parse_speakers_list(args):
    if len(args.speakers_list) == 0:
        return []
    speakers_list = args.speakers_list.replace(" ", "")
    speakers_list = speakers_list.split(",")
    return speakers_list
        
def parse_disease(args):
    if len(args.disease) == 0:
        return []
    disease = args.disease.replace(" ", "")
    disease = disease.split(",")
    return disease
    
def parse_gender(args):
    if len(args.gender) == 0:
        return []
    gender = args.gender.replace(" ", "")
    gender = gender.split(",")
    return gender
    
def control_diseases(ALL_DISEASES, diseases_list):
    ret = True
    for dis in diseases_list:
        if not dis in ALL_DISEASES.keys():
            print('Error! ', dis, ' is not allowed in diseases list')
            ret = False
    return ret

def control_gender(gender_list):
    ret = True
    i = 0
    for gen in gender_list:
        if gen == 'M':
            gender_list[i] = 'male'
        elif gen == 'F':
            gender_list[i] = 'female'
        else:
            print('Error! ', gen, ' is not a gender')
            ret = False
        i += 1
    return ret

######################## MAIN ###############################
# control inputs
if len(sys.argv) < 1:
    errorMesg()
    exit()
# as flag you can pass:
# --disease: a specific or set of disease to be analyse
# --speakers_list: speakers id list of speakers that we want analyse
# --random_train_test: random wave file selection for test and train set
# --database: database path
# --gender: gender M or F or both
########################Flag parsing###############################
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--disease",
    type=str,
    default='ALL',
    help="""Specify the disease that you want analyse.
         All the possible diseases are: 
          - ALS: Amyotrophic lateral sclerosis
          - PRK: Parkinson Disease
          - HC: Huntington's Chorea
          - STR: Stroke
          - MS: Multiple Sclerosis
          - TBI: Traumatic Brain Injury
          - MD: Myotonic dystrophy
          - ATX: Ataxia
          - NEU: Neuropathy
          - OTH: other
          - ALL: all diseases 
         You can pass more then one disease in this way: --disease \'ALS,PRK\'""",
)

parser.add_argument(
    "--speakers_list",
    type=str,
    default='ALL',
    help="""Specify the ID speakers list to analyse. To use all speakers type ALL.
         You can pass more then one user in this way: --speakers_list \'201,304\'
         If you use this flag combined with --disease, it take into account 
         just speakers with that specific disease.""",
)

parser.add_argument(
    "--random_train_test",
    type=int,
    default=0,
    help="Specify if the generation of Train and Test set is made randomly: [0, 1]",
)

parser.add_argument(
    "--database",
    type=str,
    default='./',
    help="Specify IDEA database path",
)

parser.add_argument(
    "--gender",
    type=str,
    default='M,F',
    help="Specify the gender that you want analyse: \'M\', \'F\' or \'M,F\'",
)

args = parser.parse_args()

######################Parsing arguments#######################
diseases_list = parse_disease(args)
speakers_list = parse_speakers_list(args)
dbDir = parse_database(args)
gender = parse_gender(args)
random_train_test = parse_random_train_test(args)

if not control_diseases(ALL_DISEASES, diseases_list):
    print('Control --diseases flag...')
    exit()

if not control_gender(gender):
    print('Control --gender flag...')
    exit()

# if speakers list contains ALL it must be alone
if 'ALL' in speakers_list and len(speakers_list)>1:
    print('Error! if you want use ALL in speakers list, it must be alone')
    print('Control --speakers_list flag...')
    exit()

# if ALL is in diseases_list put all diseases in the list
if 'ALL' in diseases_list:
    diseases_list = []
    for dis in ALL_DISEASES.keys():
        if not dis=='ALL':
            diseases_list.append(dis)

#print(diseases_list)

# create speakers list
listOfSpeaker = {}
if os.path.isdir(dbDir):
    # first level: DISEASES
    for disease in os.listdir(dbDir):
        # control if disease is in the list
        if disease in {k: ALL_DISEASES[k] for k in diseases_list}.values():
            # second level: GENDER
            for gend in os.listdir(dbDir+'/'+disease):
                # control if the gender is in the list
                if gend in gender:
                    # third level: SPEAKER
                    for spIdCode in os.listdir(dbDir+'/'+disease+'/'+gend):
                        # control if ID is in the list or ALL is in the list
                        if spIdCode in speakers_list or 'ALL' in speakers_list:
                            #print('Dis: ', disease)
                            #print('Gend: ', gend)
                            #print('Id: ', spIdCode)
                            if spIdCode in speakers_list:
                                speakers_list.remove(spIdCode)
                            parseSpeaker(listOfSpeaker, spIdCode, dbDir, disease, gend)
                    
# control if some ID were absent in the database
if len(speakers_list) > 0 and not 'ALL' in speakers_list:
    print('Warning! Some speakers are not present within the database: ', speakers_list)
    
# control if listOfSpeaker is empty
if len(listOfSpeaker) == 0:
    print("We did not find any speaker with your requirements...")
    print('The program is shutting down...')
    exit()

print('Tot number of speakers: ', len(listOfSpeaker))

# load phonemes from csv
if os.path.exists(dbDir+'/wordToPhonemes.csv'):
    data = pandas.read_csv(dbDir+'/wordToPhonemes.csv')
    data = data.fillna('')
    phonemes = Phonemes()
    for col in data.keys()[1:]:
        for row in data[col]:
            if row != '':
                phonemes.insert(row)

    wordsList = {}
    for (i,w) in enumerate(data['word']):
        phonList = []
        for col in data.keys()[1:]:
            ph = data[col][i]
            if ph != '':
                phonList.append(ph)
        wordsList[w] = phonList
else:
    print('Error! '+dbDir+'/wordToPhonemes.csv does not exist')
    exit()

# generate files
spALL = Speaker(-1, dbDir, '', '')
for sp in listOfSpeaker.keys():
    spALL.mergeSpeakers(listOfSpeaker[sp])
generateKaldiFile(spALL.getTestAndTrainSet(random_train_test), './', wordsList, list(phonemes.listOfPhonemes.keys()))

if random_train_test:
    print('Train and Test set has been generated RANDOMLY')
else:
    print('Train and Test set has NOT been generated RANDOMLY')