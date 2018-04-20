#Ethan Hartzell Machine Learning
#3.1 run the default algorithms and plot accuracy
#note: you must be in bash and run this: export CLASSPATH=/r/aiml/ml-software/weka-3-6-11/weka.jar:$CLASSPATH
import glob
import os, sys
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
#find the data files
data = glob.glob("data/*.arff")

#a function to fine matching training and test sets based on number of features in filename
def match_sets(tset,num):
        train = ""
        test = ""
        for file in data:
                if "Train" in file:
                        if str(num) in file:
                                train = file
                if "Test" in file:
                        if str(num) in file:
                                test = file
        return [train,test]

#this function takes the algorithm command and the number of features
#and it returns the accuracy in a tuple with the number of features 
def experiment(num,algo):
        train = ""
        test = ""
        tset = [train,test]
        tset = match_sets(tset,num)
        #calls weka and runs the specified data files
        tmp = os.popen(algo + "-t " + tset[0] + " -T " + tset[1]).read().split()
        td = False #just a bool to help keep track of where in the output we are
        percent = ""
        #reads the output and finds the accuracy on the test set
        for i,word in enumerate(tmp):
                if word == "test":
                        td = True
                if td:
                        if word == "Correctly":
                                percent = tmp[i+4]
        return (num,float(percent))
#does the same as above but with the training and test sets already specified for 3.2
def exp(algo,tset):
        #calls weka and runs the specified data files
        tmp = os.popen(algo + "-t " + tset[0] + " -T " + tset[1]).read().split()
        td = False #just a bool to help keep track of where in the output we are
        percent = ""
        #reads the output and finds the accuracy on the test set
        for i,word in enumerate(tmp):
                if word == "test":
                        td = True
                if td:
                        if word == "Correctly":
                                percent = tmp[i+4]
        #detect the number on the end of the file name (won't be more than three digits)
        num = 0
        if tset[0][-8] != '_':
                num = int(tset[0][-8] + tset[0][-7] + tset[0][-6])
        else:
                num = int(tset[0][-7]+tset[0][-6])
        return (num,float(percent))

#here we run the experiment of 3.1 on each condition (# of features) and record the results
results = []
#first we use a decision tree algorithm
algo = "java weka.classifiers.trees.J48 "
results.append(experiment(14,algo))
results.append(experiment(24,algo))
results.append(experiment(34,algo))
results.append(experiment(44,algo))
results.append(experiment(54,algo))
results.append(experiment(64,algo))
results.append(experiment(74,algo))
results.append(experiment(84,algo))
results.append(experiment(94,algo))
print results
n_results = []
#next we use the nearest neighbors algorithm
algo = "java weka.classifiers.lazy.IBk "
n_results.append(experiment(14,algo))
n_results.append(experiment(24,algo))
n_results.append(experiment(34,algo))
n_results.append(experiment(44,algo))
n_results.append(experiment(54,algo))
n_results.append(experiment(64,algo))
n_results.append(experiment(74,algo))
n_results.append(experiment(84,algo))
n_results.append(experiment(94,algo))

print n_results
#we convert the results to a form amenable to matplotlib
x = []
y1= []
print results[0]
print results[1]
for i,j in results:
        x.append(i)
        y1.append(j)
print x
print y1
y2 = []
for i,j in n_results:
        y2.append(j)
#we plot the results, they pop up in a separate window. this graph is also saved in the zip file
plt.plot(x,y1,"r", x,y2)
plt.title("IBk (blue) vs. J48 (red)")
plt.xlabel("number of features")
plt.ylabel("accuracy out of 100%")
plt.show()

#here we begin 3.2 

#this class can read an arff file and have its data manipulated and then have another rewritten
class arff_file:
        def __init__(self,filename):
                self.relation = ""
                self.attributes = []
                self.data = []
                self.labeled_data = 
                self.file = filename
                f = open(filename)
                tmp = f.readlines()
                data = False
                #store the relevent parts of the file
                for line in tmp:
                        ls = line.split()
                        for i, word in enumerate(ls):
                                if word == "@Relation":
                                        self.relation = ls[i+1]
                                if word == "@ATTRIBUTE":
                                        self.attributes.append(line)
                                        
                                if word == "@DATA":
                                        data = True
                                        continue
                        if data:
                                if line != "@DATA":
                                        self.data.append(line)
                f.close()

        def write_file(self,new_file_name):
                #write a new arff file based on whats recorded in the class
                f = open(new_file_name,'w')
                f.write("@Relation " + self.relation + "\n")
                f.write("\n")
                for at in self.attributes:
                        f.write(at)
                f.write("\n")
                f.write("@DATA\n")
                for example in self.data:
                        if example != "@DATA":
                                f.write(example)
                f.write("\n")
                f.close()
        def get_data(self):
                return self.data
        def replace_data(self,nd):
                self.data = nd

train14 = arff_file("data/EEGTrainingData_14.arff")
d = train14.get_data()
for i in d:
        if "@DATA" in i:
                d.remove(i)
import random
#randomizes the data and runs a trial on each size for both algorithms and returns the accuracies
def rando_trial():
        random.shuffle(d)
        os.system("rm *.arff")
        for i in range(50,550,50):
                new_arr = []
                for count, line in enumerate(d):
                        if count < i:
                                new_arr.append(line)
                train14.replace_data(new_arr)
                nfn = "EEGTrainingData_14_"
                train14.write_file(nfn + str(i) +".arff")
        training_files = glob.glob("*.arff")
        test_file = "data/EEGTestingData_14.arff"
        t14_j48_res=[]
        t14_ibk_res=[]
        for file in training_files:
                algo = "java weka.classifiers.lazy.IBk "
                t14_ibk_res.append(exp(algo,(file,test_file)))
                algo = "java weka.classifiers.trees.J48 "
                t14_j48_res.append(exp(algo,(file,test_file)))
        return (t14_j48_res,t14_ibk_res)
t14_j48_avg = defaultdict(float) #holds the means
t14_ibk_avg = defaultdict(float) #holds the means
j48_scores = defaultdict(list) #holds all the trial data
ibk_scores = defaultdict(list)
#run 10 trials with randomized data
for k in range(10):
        j = rando_trial()
        for i in j[0]:
                #t14_j48_avg[i[0]] += i[1]
                j48_scores[i[0]].append(i[1])
        for i in j[1]:
                #t14_ibk_avg[i[0]] += i[1]
                ibk_scores[i[0]].append(i[1])
#finish calculating the means
for i, v in j48_scores.iteritems():
        for val in v:
                t14_j48_avg[i] += val
        t14_j48_avg[i] = t14_j48_avg[i]/10.0
for i, v in ibk_scores.iteritems():
        for val in v:
                t14_ibk_avg[i] += val
        t14_ibk_avg[i] = t14_ibk_avg[i]/10.0
t14_j48_sd = defaultdict(float) #holds the standard deviations
t14_ibk_sd = defaultdict(float)
#calculate the sd
print j48_scores
for i,v in j48_scores.iteritems():
        for val in v:
                t14_j48_sd[i] += (val-t14_j48_avg[i])**2
for i,v in t14_j48_sd.iteritems():
        t14_j48_sd[i] = (val /10.0) **0.5

for i,v in ibk_scores.iteritems():
        for val in v:
                t14_ibk_sd[i] += (val-t14_ibk_avg[i])**2
        
for i,v in t14_ibk_sd.iteritems():
        t14_ibk_sd[i] = (val/10.0)**0.5
print t14_j48_sd
print t14_j48_avg
print "^ SD and Average"

#now lets do it all again for the 54 feature data
print "running on 54 now"
train54 = arff_file("data/EEGTrainingData_54.arff")
os.system("rm *arff")
d = train54.get_data()
for i in d:
        if "@DATA" in i:
                d.remove(i)
#randomizes the data and runs a trial on each size for both algorithms and returns the accuracies
def random_trial():
        random.shuffle(d)
        for i in range(50,550,50):
                new_arr = []
                for count, line in enumerate(d):
                        if count < i:
                                new_arr.append(line)
                train54.replace_data(new_arr)
                nfn = "EEGTrainingData_54_"
                train54.write_file(nfn + str(i) +".arff")
        training_files = glob.glob("*.arff")
        test_file = "data/EEGTestingData_54.arff"
        t14_j48_res=[]
        t14_ibk_res=[]
        for file in training_files:
                algo = "java weka.classifiers.lazy.IBk "
                t14_ibk_res.append(exp(algo,(file,test_file)))
                algo = "java weka.classifiers.trees.J48 "
                t14_j48_res.append(exp(algo,(file,test_file)))
        return (t14_j48_res,t14_ibk_res)
t54_j48_avg = defaultdict(float) #holds the means
t54_ibk_avg = defaultdict(float) #holds the means
j48_scores54 = defaultdict(list) #holds all the trial data
ibk_scores54 = defaultdict(list)
#run 10 trials with randomized data
#run 10 trials with randomized data
for k in range(10):
        j = random_trial()
        for i in j[0]:
                t54_j48_avg[i[0]] += i[1]
                j48_scores54[i[0]].append(i[1])
        for i in j[1]:
                t54_ibk_avg[i[0]] += i[1]
                ibk_scores54[i[0]].append(i[1])
#finish calculating the means
for i, v in j48_scores54.iteritems():
        for val in v:
                t54_j48_avg[i] += val
        t54_j48_avg[i] = t54_j48_avg[i]/20.0
for i, v in ibk_scores54.iteritems():
        for val in v:
                t54_ibk_avg[i] += val
        t54_ibk_avg[i] = t54_ibk_avg[i]/20.0
t54_j48_sd = defaultdict(float) #holds the standard deviations
t54_ibk_sd = defaultdict(float)
#calculate the sd
print j48_scores54
for i,v in j48_scores54.iteritems():
        for val in v:
                t54_j48_sd[i] += (val-t54_j48_avg[i])**2
for i,v in t54_j48_sd.iteritems():
        t54_j48_sd[i] = (val /10.0) **0.5

for i,v in ibk_scores54.iteritems():
        for val in v:
                t54_ibk_sd[i] += (val-t54_ibk_avg[i])**2
        
for i,v in t54_ibk_sd.iteritems():
        t54_ibk_sd[i] = (val/10.0)**0.5
print t54_j48_sd
print t54_j48_avg
print "^ SD and Average"
plt.figure()
x = []
j = t54_j48_avg.keys()
j.sort()
for i in j:
        x.append(i)
y1 = [] #t14 j48 results
y1err = []
y2 = [] #t14 ibk results
y2err = []
y3 = [] #t54 j48 results
y3err = []
y4 = [] #t53 ibk results
y4err = []
print x
for key in x:
        y1.append(t14_j48_avg[key])
        y2.append(t14_ibk_avg[key])
        y3.append(t54_j48_avg[key])
        y4.append(t54_ibk_avg[key])
        y1err.append(t14_j48_sd[key])
        y2err.append(t14_ibk_sd[key])
        y3err.append(t54_j48_sd[key])
        y4err.append(t54_ibk_sd[key])
plt.errorbar(x,y1,y1err)
plt.errorbar(x,y2,y2err)
plt.errorbar(x,y3,y3err)
plt.errorbar(x,y4,y4err)
plt.show()