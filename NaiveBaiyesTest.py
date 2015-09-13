from numpy import *
def loadSet():
    
    f=open('nursery.txt')
    count=0
    listVal=zeros((12958,8))
    classLabels=[None]*12958
    while count<12959:
        line=f.readline()
        line=line[0:-1]
        strings=(line.split(','))
        listVal[count,:]=(linetoVec(line))
        classLabels[count]=returnClass(line)
        count=count+1
        if count==12958:
            break
    f.close()
    return listVal,classLabels    

def linetoVec(line):
    listVal=zeros((1,8))
    
    count=0
    strings=(line.split(','))
    if strings[0]=='usual':
        listVal[count,0]=1
    elif strings[0]=='pretentious':
        listVal[count,0]=2
    elif strings[0]=='great_pret':
        listVal[count,0]=3

    if strings[1]=='proper':
        listVal[count,1]=1
    elif strings[1]=='less_proper':
        listVal[count,1]=2
    elif strings[1]=='improper':
        listVal[count,1]=3
    elif strings[1]=='critical':
        listVal[count,1]=4
    elif strings[1]=='very_crit':
        listVal[count,1]=5

    if strings[2]=='complete':
        listVal[count,2]=1
    elif strings[2]=='completed':
        listVal[count,2]=2
    elif strings[2]=='incomplete':
        listVal[count,2]=3
    elif strings[2]=='foster':
        listVal[count,2]=4

    if strings[3]=='1':
        listVal[count,3]=1
    elif strings[3]=='2':
        listVal[count,3]=2
    elif strings[3]=='3':
        listVal[count,3]=3
    elif strings[3]=='more':
        listVal[count,3]=4

    if strings[4]=='convenient':
        listVal[count,4]=1
    elif strings[4]=='less_conv':
        listVal[count,4]=2
    elif strings[4]=='critical':
        listVal[count,4]=3

    if strings[5]=='convenient':
        listVal[count,5]=1
    elif strings[5]=='inconv':
        listVal[count,5]=2

    if strings[6]=='nonprob':
        listVal[count,6]=1
    elif strings[6]=='slightly_prob':
        listVal[count,6]=2
    elif strings[6]=='problematic':
        listVal[count,6]=3

    if strings[7]=='recommended':
        listVal[count,7]=1
    elif strings[7]=='priority':
        listVal[count,7]=2
    elif strings[7]=='not_recom':
        listVal[count,7]=3

    if strings[8]=='not_recom':
        labelOfClass=1
    elif strings[8]=='recommend':
        labelOfClass=2
    if strings[8]=='very_recom':
        labelOfClass=3
    if strings[8]=='priority':
        labelOfClass=4
    if strings[8]=='spec_prior':
        labelOfClass=5

    return listVal

def returnClass(line):
    strings=(line.split(','))
    if strings[8]=='not_recom':
        labelOfClass=1
    elif strings[8]=='recommend':
        labelOfClass=2
    if strings[8]=='very_recom':
        labelOfClass=3
    if strings[8]=='priority':
        labelOfClass=4
    if strings[8]=='spec_prior':
        labelOfClass=5
    return labelOfClass
    
def classify(firstAttr,secondAttr,thirdAttr,fourthAttr,fifthAttr,sixthAttr,seventhAttr,eigthAttr,classProbList,listVal):
    classvals=log(firstAttr[listVal[0]-1,:])+log(secondAttr[listVal[1]-1,:])+log(thirdAttr[listVal[2]-1,:])+log(fourthAttr[listVal[3]-1,:])+log(fifthAttr[listVal[4]-1,:])+log(sixthAttr[listVal[5]-1,:])+log(seventhAttr[listVal[6]-1,:])+log(eigthAttr[listVal[7]-1,:])+log(classProbList[:])
    return classvals    

def trainingFunc(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numAttributes=len(trainMatrix[0])

    class1Prob=(float)(trainCategory.count(1))/(float)(numTrainDocs)
    class2Prob=(float)(trainCategory.count(2))/(float)(numTrainDocs)
    class3Prob=(float)(trainCategory.count(3))/(float)(numTrainDocs)
    class4Prob=(float)(trainCategory.count(4))/(float)(numTrainDocs)
    class5Prob=(float)(trainCategory.count(5))/(float)(numTrainDocs)
    
    classProbList=[class1Prob,class2Prob,class3Prob,class4Prob,class5Prob]
    firstAttr=[[1.0]*3 for i in range(5)]
    secondAttr=[[1.0]*5 for i in range(5)]
    thirdAttr=[[1.0]*4 for i in range(5)]
    fourthAttr=[[1.0]*4 for i in range(5)]
    fifthAttr=[[1.0]*3 for i in range(5)]
    sixthAttr=[[1.0]*2 for i in range(5)]
    seventhAttr=[[1.0]*3 for i in range(5)]
    eigthAttr=[[1.0]*3 for i in range(5)]

    for i in range(numTrainDocs):
      #  print trainMatrix[i][0],trainMatrix[i][1],trainMatrix[i][2],trainMatrix[i][3],trainMatrix[i][4],trainMatrix[i][5],trainMatrix[i][6],trainMatrix[i][7]    
        firstAttr[int(trainCategory[i])-1][int(trainMatrix[i][0])-1]+=1.0
        secondAttr[int(trainCategory[i])-1][int(trainMatrix[i][1])-1]+=1.0
        thirdAttr[int(trainCategory[i])-1][int(trainMatrix[i][2])-1]+=1.0
        fourthAttr[int(trainCategory[i])-1][int(trainMatrix[i][3])-1]+=1.0
        fifthAttr[int(trainCategory[i])-1][int(trainMatrix[i][4])-1]+=1.0
        sixthAttr[int(trainCategory[i])-1][int(trainMatrix[i][5])-1]+=1.0
        seventhAttr[int(trainCategory[i])-1][int(trainMatrix[i][6])-1]+=1.0
        eigthAttr[int(trainCategory[i])-1][int(trainMatrix[i][7])-1]+=1.0

    class1count=(float)(trainCategory.count(1))
    class2count=(float)(trainCategory.count(2))
    class3count=(float)(trainCategory.count(3))
    class4count=(float)(trainCategory.count(4))
    class5count=(float)(trainCategory.count(5))

    firstAttr=array(firstAttr).T
    secondAttr=array(secondAttr).T
    thirdAttr=array(thirdAttr).T
    fourthAttr=array(fourthAttr).T
    fifthAttr=array(fifthAttr).T
    sixthAttr=array(sixthAttr).T
    seventhAttr=array(seventhAttr).T
    eigthAttr=array(eigthAttr).T
    
    firstAttr[:,0]=firstAttr[:,0]/(class1count+4.0)
    firstAttr[:,1]=firstAttr[:,1]/(class2count+4.0)
    firstAttr[:,2]=firstAttr[:,2]/(class3count+4.0)
    firstAttr[:,3]=firstAttr[:,3]/(class4count+4.0)
    firstAttr[:,4]=firstAttr[:,4]/(class5count+4.0)
   

    secondAttr[:,0]=secondAttr[:,0]/(class1count+6.0)
    secondAttr[:,1]=secondAttr[:,1]/(class2count+6.0)
    secondAttr[:,2]=secondAttr[:,2]/(class3count+6.0)
    secondAttr[:,3]=secondAttr[:,3]/(class4count+6.0)
    secondAttr[:,4]=secondAttr[:,4]/(class5count+6.0)

    thirdAttr[:,0]=thirdAttr[:,0]/(class1count+5.0)
    thirdAttr[:,1]=thirdAttr[:,1]/(class2count+5.0)
    thirdAttr[:,2]=thirdAttr[:,2]/(class3count+5.0)
    thirdAttr[:,3]=thirdAttr[:,3]/(class4count+5.0)
    thirdAttr[:,4]=thirdAttr[:,4]/(class5count+5.0)

    fourthAttr[:,0]=fourthAttr[:,0]/(class1count+5.0)
    fourthAttr[:,1]=fourthAttr[:,1]/(class2count+5.0)
    fourthAttr[:,2]=fourthAttr[:,2]/(class3count+5.0)
    fourthAttr[:,3]=fourthAttr[:,3]/(class4count+5.0)
    fourthAttr[:,4]=fourthAttr[:,4]/(class5count+5.0)

    fifthAttr[:,0]=fifthAttr[:,0]/(class1count+4.0)
    fifthAttr[:,1]=fifthAttr[:,1]/(class2count+4.0)
    fifthAttr[:,2]=fifthAttr[:,2]/(class3count+4.0)
    fifthAttr[:,3]=fifthAttr[:,3]/(class4count+4.0)
    fifthAttr[:,4]=fifthAttr[:,4]/(class5count+4.0)

    sixthAttr[:,0]=sixthAttr[:,0]/(class1count+3.0)
    sixthAttr[:,1]=sixthAttr[:,1]/(class2count+3.0)
    sixthAttr[:,2]=sixthAttr[:,2]/(class3count+3.0)
    sixthAttr[:,3]=sixthAttr[:,3]/(class4count+3.0)
    sixthAttr[:,4]=sixthAttr[:,4]/(class5count+3.0)

    seventhAttr[:,0]=seventhAttr[:,0]/(class1count+4.0)
    seventhAttr[:,1]=seventhAttr[:,1]/(class2count+4.0)
    seventhAttr[:,2]=seventhAttr[:,2]/(class3count+4.0)
    seventhAttr[:,3]=seventhAttr[:,3]/(class4count+4.0)
    seventhAttr[:,4]=seventhAttr[:,4]/(class5count+4.0)

    eigthAttr[:,0]=eigthAttr[:,0]/(class1count+4.0)
    eigthAttr[:,1]=eigthAttr[:,1]/(class2count+4.0)
    eigthAttr[:,2]=eigthAttr[:,2]/(class3count+4.0)
    eigthAttr[:,3]=eigthAttr[:,3]/(class4count+4.0)
    eigthAttr[:,4]=eigthAttr[:,4]/(class5count+4.0)

    print class1count,class2count,class3count,class4count,class5count
    return firstAttr,secondAttr,thirdAttr,fourthAttr,fifthAttr,sixthAttr,seventhAttr,eigthAttr,classProbList
                
def testing():
    [listVal,classLabels]=loadSet()
    [firstAttr,secondAttr,thirdAttr,fourthAttr,fifthAttr,sixthAttr,seventhAttr,eigthAttr,classProbList]=trainingFunc(listVal,classLabels)
    [listVal]=linetoVec('usual,proper,complete,1,convenient,convenient,nonprob,not_recom,not_recom')
    classValue=classify(firstAttr,secondAttr,thirdAttr,fourthAttr,fifthAttr,sixthAttr,seventhAttr,eigthAttr,classProbList,listVal)
    print classValue.argmax()+1
    return listVal,classLabels,firstAttr,secondAttr,thirdAttr,fourthAttr,fifthAttr,sixthAttr,seventhAttr,eigthAttr,classProbList,classValue
def crossValidation():
    [listVal,classLabels]=loadSet()
    [firstAttr,secondAttr,thirdAttr,fourthAttr,fifthAttr,sixthAttr,seventhAttr,eigthAttr,classProbList]=trainingFunc(listVal,classLabels)
    f=open('nursery.txt')
    count=0
    error=0.0
    while count<12959:
        line=f.readline()
        line=line[0:-1]
        
        [testVec]=linetoVec(line)
        labelOfClass=returnClass(line)
        classValue=classify(firstAttr,secondAttr,thirdAttr,fourthAttr,fifthAttr,sixthAttr,seventhAttr,eigthAttr,classProbList,testVec)
        if (classValue.argmax()+1)!=labelOfClass:
            error+=1.0
        count+=1
    print error/12959.0*100.0
