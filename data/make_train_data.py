import sys
import commands
import subprocess

def cmd(cmd):
	return commands.getoutput(cmd)
#	p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#	p.wait()
#	stdout, stderr = p.communicate()
#	return stdout.rstrip()

#labels
dirs = cmd("ls "+sys.argv[1])
labels = dirs.splitlines()

#make directries
cmd("mkdir images")

#copy images and make train.txt
pwd = cmd('pwd')
imageDir = pwd+"/images"
train = open('train.txt','w')
train_lstm = open('train_lstm.tsv','w')
test = open('test.txt','w')
labelsTxt = open('labels.txt','w')

classNo=0
cnt = 0
#label = labels[classNo]
for label in labels:
	workdir = pwd+"/"+sys.argv[1]+"/"+label
	imageFiles = cmd("ls "+workdir+"/*.jpg")
	images = imageFiles.splitlines()
	print(label)
	labelsTxt.write(label+"\n")
	startCnt=cnt
	length = len(images)
	for image in images:
		imagepath = imageDir+"/image%07d" %cnt +".jpg"
		cmd("cp "+image+" "+imagepath)
		if cnt-startCnt < length*0.75:
			train.write(imagepath+" %d\n" % classNo)
			train_lstm.write(imagepath+"\t%s\n" % label)
		else:
			test.write(imagepath+" %d\n" % classNo)
		cnt += 1
	
	classNo += 1

train.close()
test.close()
train_lstm.close()
labelsTxt.close()
