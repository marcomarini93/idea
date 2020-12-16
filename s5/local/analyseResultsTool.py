#!/usr/bin/env python3
import os
import pandas
import math
import sys
import random


"""
This program evaluate WER for each speaker.

It takes as input the exp folder and parsing the files 
contained in exp/train/{tri3,dnn_fbank}/decode_test/scoring_kaldi/wer_details
generates generates 2 files:
 - speakerStatistics_tri3
 - speakerStatistics_dnn_fbank
that reassume some statistics for each speaker
"""

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
	def __init__(self, id):
		self.id = id
	def setPath(self, path):
		self.path = path
	def setWord(self, word):
		self.word = word
	def setStart(self, start):
		self.start = start
	def setEnd(self, end):
		self.end = end
	def setSpk(self, spk):
		self.spk = spk
	def setNotes(self, notes):
		self.notes = notes
	def setDuration(self, duration):
		self.duration = duration
	def getVoice(self):
		return self.end-self.start
	def setTrain(self, train):
		self.train = train
	def haveNotes(self):
		#print(self.notes)
		return sum(self.notes)>0

class Speaker:
	def __init__(self, id):
		self.id = id
		self.words = []
		self.records = []
		self.emptyFiles = 0
		self.TestAndTrain = {}
		self.TestAndTrain['test'] = []
		self.TestAndTrain['train'] = []
		self.WER = 0
		self.sub = 0
		self.ins = 0
		self.delet = 0
	def getTestAndTrainSet(self):
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
			ret += r.getVoice()
		return ret
	def getTotHoursOfTest(self):
		ret = 0
		for r in self.TestAndTrain['test']:
			ret += r.duration
		return ret
	def getVoiceHoursOfTest(self):
		ret = 0
		for r in self.TestAndTrain['test']:
			ret += r.getVoice()
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
		#		  1/0 -> substitution
		#         1/0 -> repetition
		#         1/0 -> corrupted
		#         1/0 -> background noise
		#         1/0 -> not usable
		#         1/0 -> word splitted
		#         1/0 -> general notes 
		#		  1/0 -> without notes]
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
		#	- duration of entier rec
		#	- usable rec (by def is 1 because we assume that every rec are usable)
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
							#print("	-> "+interval.mark+", "+str(tmpStart)+", "+str(tmpEnd))
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
			ret -1
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


def errorMesg():
	print("""Error! 
		The program should be runned with these arguments:
		genTestAntTrain2o1.py <database folder> <list of spks id>""")

def parseSpeaker(listOfSpeaker, spIdCode, dbDir):
	spkDir = dbDir+'/'+spIdCode
	if os.path.exists(spkDir):
		if os.path.isdir(spkDir):
			#listOfSpeaker = {}
			#speakerKeysValues = []
			#for usId in os.listdir(os.getcwd()):
			#	if ( os.path.isdir(os.getcwd()+'/'+usId)):
			listOfSpeaker[spIdCode] = Speaker(spIdCode, dbDir)
			#speakerKeysValues += [str(usId)]
			sp1 = listOfSpeaker[str(spIdCode)]
			print('User id: '+str(sp1.id))
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
		exit()

def isTrainOrTest(path):
	if 'train' in path:
		return True
	else:
		return False

def parseText(path, listOfRec):
	# extract word from text file
	with open(path, 'r') as file:
		lines = file.readlines()
		for l in lines:
			info = l.split()
			idSplitted = info[0].split("_")
			id = idSplitted[0]+'_'+idSplitted[1]+'_'+idSplitted[2]+'_'+idSplitted[3]
			if not id in listOfRec:
				#print("nuovo: "+id)
				rec = Record(id)
				rec.setWord(info[1])
				rec.setTrain(isTrainOrTest(path))
				listOfRec[id] = rec
			else:
				rec = listOfRec[id]
				rec.setWord(info[1])
	#print("Text: "+str(len(listOfRec.keys())))

def parseWavscp(path, listOfRec):
	# extract path from wav.scp file
	with open(path, 'r') as file:
		lines = file.readlines()
		for l in lines:
			info = l.split()
			if not info[0] in listOfRec:
				rec = Record(info[0])
				rec.setPath(info[1])
				rec.setTrain(isTrainOrTest(path))
				listOfRec[info[0]] = rec
			else:
				rec = listOfRec[info[0]]
				rec.setPath(info[1])
	#print("Wavscp: "+str(len(listOfRec.keys())))

def parseUtt2spk(path, listOfRec):
	# extract speaker from utt2spk file
	with open(path, 'r') as file:
		lines = file.readlines()
		for l in lines:
			info = l.split()
			idSplitted = info[0].split("_")
			id = idSplitted[0]+'_'+idSplitted[1]+'_'+idSplitted[2]+'_'+idSplitted[3]
			if not id in listOfRec:
				rec = Record(id)
				rec.setSpk(info[1])
				rec.setTrain(isTrainOrTest(path))
				listOfRec[id] = rec
			else:
				rec = listOfRec[id]
				rec.setSpk(info[1])
	#print("Utt2spk: "+str(len(listOfRec.keys())))

def parseSegments(path, listOfRec):
	# extract start, end and voice  from utt2spk file
	with open(path, 'r') as file:
		lines = file.readlines()
		for l in lines:
			info = l.split()
			idSplitted = info[0].split("_")
			id = idSplitted[0]+'_'+idSplitted[1]+'_'+idSplitted[2]+'_'+idSplitted[3]
			if not id in listOfRec:
				rec = Record(id)
				rec.setStart(float(info[2]))
				rec.setEnd(float(info[3]))
				rec.setTrain(isTrainOrTest(path))
				listOfRec[id] = rec
			else:
				rec = listOfRec[id]
				rec.setStart(float(info[2]))
				rec.setEnd(float(info[3]))
	#print("segments: "+str(len(listOfRec.keys())))

def parsePerspkFile(path, listOfSpeaker):
	if os.path.isfile(path):
		with open(path, 'r') as file:
			lines = file.readlines()
			for l in lines:
				info = l.split()
				if info[1]=="sys" and info[0] in listOfSpeaker:
					spk = listOfSpeaker[info[0]]
					spk.sub = info[5]
					spk.ins = info[6]
					spk.delet = info[7]
					spk.WER = info[8]
	else:
		print("Error: file "+str(path)+" does not exist")
		exit()
	
# main
# find folder data
# for each test/train folder parsing files 
# - text
# - wav.scp
# - utt2spk
# - segments

if not len(sys.argv)==2:
	printf('Error: arguments error')
	printf('e.g. analyseResultsTool.py exp_1')
	exit()

trainPath = "./data/train"
testPath = "./data/test"
listOfRec = {}
for path in [trainPath, testPath]:
	if os.path.isdir(path):
		# wav.scp parsing
		parseWavscp(path+"/wav.scp", listOfRec)
		# test parsing
		parseText(path+"/text", listOfRec)
		# segments parsing
		parseSegments(path+"/segments", listOfRec)
		# utt2spk parsing
		parseUtt2spk(path+"/utt2spk", listOfRec)
	else:
		print("Error: "+path+"is not a directory")
		exit()
print("rec:")
#print(listOfRec.keys())
print("tot rec: "+str(len(listOfRec)))
trainRec = 0
testRec = 0
for r in listOfRec.values():
	if r.train:
		trainRec+=1
	else:
		testRec+=1
print("train: "+str(trainRec))
print("test: "+str(testRec))
# generate list of speakers
listOfSpeaker = {}
for r in listOfRec.values():
	spkId = r.spk
	if not spkId in listOfSpeaker:
		listOfSpeaker[spkId] = Speaker(spkId)
	spk = listOfSpeaker[spkId]
	if r.train:
		spk.TestAndTrain['train'].append(r)
	else:
		spk.TestAndTrain['test'].append(r)
		
print("speakers: "+str(len(listOfSpeaker)))	
for s in listOfSpeaker.values():
	print("spk: "+str(s.id)+", train: "+str(len(s.TestAndTrain['train']))+", sec: "+str(s.getVoiceHoursOfTrain())+", test: "+str(len(s.TestAndTrain['test']))+", sec: "+str(s.getVoiceHoursOfTest()))

# retrive info from per_spk file
for model in ["dnn_fbank","tri3"]:
	path = "./"+sys.argv[1]+"/train/"+model+"/decode_test/scoring_kaldi/wer_details/per_spk"
	parsePerspkFile(path, listOfSpeaker)

	# generate file with data
	outFile = os.getcwd()+"/speakerStatistics_"+model+".txt"
	with open(outFile, 'w') as f:
		f.write("Speaker\tWER\tTrain rec\t% train\tTrain sec\tTest rec\tTest sec\tOverall train\tSub\tIns\tDel\n")
		for s in listOfSpeaker.values():
			f.write( str(s.id) + "\t" + \
					str(s.WER) + "\t" + \
					str(len(s.TestAndTrain['train'])) + "\t" + \
					str(len(s.TestAndTrain['train'])*100/(len(s.TestAndTrain['train'])+len(s.TestAndTrain['test']))) + "\t" + \
					str(s.getVoiceHoursOfTrain()) + "\t" + \
					str(len(s.TestAndTrain['test'])) + "\t" + \
					str(s.getVoiceHoursOfTest()) + "\t" + \
					str(len(s.TestAndTrain['train'])*100/trainRec) + "\t" + \
					str(s.sub) + "\t" + \
					str(s.ins) + "\t" + \
					str(s.delet) + \
					"\n")

