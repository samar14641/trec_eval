import math
import matplotlib.pyplot as plt
import os
import sys


kVals = [5, 10, 20, 50, 100]

def readQrel(qrelFile):
    relevant = {}
    print('reading qrel')

    with open(qrelFile, 'r') as qrel:
        line = qrel.readline()

        while line:
            split = line.strip().split()

            qID, docID, rel = split[0], split[2], split[3]

            if rel != '0':
                if qID in relevant:
                    relevant[qID].append(docID)
                else:
                    relevant[qID] = [docID]

            line = qrel.readline()

    qrel.close()

    return relevant

def readRes(resFile):
    results = {}
    print('reading res')

    with open(resFile, 'r') as res:
        line = res.readline()

        while line:
            split = line.strip().split()

            qID, docID = split[0], split[2]

            if qID in results:
                results[qID].append(docID)
            else:
                results[qID] = [docID]

            line = res.readline()
    
    res.close()

    return results

def calcNDCG(rel):
    i, dcg = 1, 0.0

    for score in rel:
        if i == 1:
            dcg += score
        else:
            dcg += score / math.log(i)

        i += 1

    return dcg

def calcCutoff(P, R, F1):
    tempP, tempR, tempF1 = {}, {}, {}

    for k in kVals:
        tempP[k] = math.fsum(P[k]) / len(P[k])
        tempR[k] = math.fsum(R[k]) / len(R[k])
        tempF1[k] = math.fsum(F1[k]) / len(F1[k])

    return tempP, tempR, tempF1

def printer(qID, ret, relDocs, relRet, avgPrec, rPrec, dcg, precCutoff, recCutoff, f1Cutoff):
    print('Query ID (num):', qID)
    print('\tRetrieved:', ret)
    print('\tRelevant:', relDocs)
    print('\tRelevant retrieved:', relRet)
    print('Avg prec (non-interpolated):', avgPrec)
    print('nDCG:', dcg)

    print('\n    k      Precision@k   Recall@k    F1-Measure@k')
    print('   ---     -----------   --------    ------------')
    for k in kVals:
        print('{0:4d} docs    {1:.4f}       {2:.4f}        {3:.4f}'.format(k, precCutoff[k], recCutoff[k], f1Cutoff[k]))
    print('R-Precision:', rPrec)

def graph(prec, rec, qID):
    print('graphing', qID)
    plt.plot(rec, prec)
    plt.title('Precision - Recall Curve for Query ' + qID)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.savefig(os.getcwd() + '\\Graphs\\' + qID + '_all')   
    plt.close()

def calc(option, relevant, results):
    totQuery, totRel, totRelRet, totRet = 0, 0, 0, 0
    avgPrec, rPrec, nDCG = [], [], []
    P, R, F1 = {}, {}, {}

    for qID in sorted(results.keys()):
        totQuery += 1
        relevanceScore = []
        totPrec, totRec = {}, {}
        sumPrec, ret, relRet, rp = 0, 0, 0, 0

        res = results[qID]

        if qID not in relevant:
            continue

        precCutQuery, recCutQuery, f1CutQuery = {}, {}, {}

        relResDocs = relevant[qID]
        numRel = len(relResDocs)
        totRel += numRel

        for doc in res:
            ret += 1
            isRelevant = False

            if doc in relResDocs:
                relRet += 1
                isRelevant = True

            precision = relRet / (ret)
            score = 0

            if isRelevant:
                sumPrec += precision
                score = 1

            relevanceScore.append(score)

            if numRel == 0:
                recall = 0
            else:
                recall = relRet / numRel

            if ret <= numRel:
                rp = relRet

            totPrec[ret] = precision
            totRec[ret] = recall

        count = ret
        finalRecall = relRet / numRel

        allPrec, allRec = [], []

        while count <= 1000:
            totPrec[count] = relRet / count
            totRec[count] = finalRecall
            allPrec.append(totPrec[count])
            allRec.append(totRec[count])
            count += 1

        precCutoff, recCutoff = [], []

        for k in kVals:
            tempP = totPrec[k]
            tempR = totRec[k]

            precCutoff.append(tempP)
            recCutoff.append(tempR)

            f1 = 0
            if tempP > 0 and tempR > 0:
                f1 = (2 * tempP * tempR) / (tempP + tempR)

            precCutQuery[k] = tempP
            recCutQuery[k] = tempR
            f1CutQuery[k] = f1

            if k in P:
                P[k].append(tempP)
            else:
                P[k] = [tempP]

            if k in R:
                R[k].append(tempR)
            else:
                R[k] = [tempR]

            if k in F1:
                F1[k].append(f1)
            else:
                F1[k] = [f1]

        avgP = 0.0
        
        if relRet != 0:
            avgP = sumPrec / numRel

        avgPrec.append(avgP)
        rPrec.append(rp / len(relResDocs))
        
        numDCG = calcNDCG(relevanceScore)
        relevanceScoreDesc = sorted(relevanceScore, reverse = True)
        dcgDesc = calcNDCG(relevanceScoreDesc)

        # graph(precCutoff, recCutoff, qID)

        tempNDCG = 0
        if dcgDesc != 0:
            tempNDCG = numDCG / dcgDesc
        
        nDCG.append(tempNDCG)
        totRelRet += relRet
        totRet += ret

        if option:
            printer(qID, ret, numRel, relRet, avgP, rp / len(relResDocs), tempNDCG, precCutQuery, recCutQuery, f1CutQuery)

    AvgPrec = math.fsum(avgPrec) / len(avgPrec)
    RPre = math.fsum(rPrec) / len(rPrec)
    NDCG = math.fsum(nDCG) / len(nDCG)

    finalP, finalR, finalF1 = calcCutoff(P, R, F1)

    printer(totQuery, totRet, totRel, totRelRet, AvgPrec, RPre, NDCG, finalP, finalR, finalF1)

def main():
    option = False
    qrelFile = None
    resFile = None

    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('Incorrect format')
        sys.exit(0)

    if len(sys.argv) == 3:
        qrelFile = sys.argv[1]
        resFile = sys.argv[2]
    else:
        option = True
        qrelFile = sys.argv[2]
        resFile = sys.argv[3]

    # print(option, qrelFile, resFile)

    relevant = readQrel(os.getcwd() + '\\Result Files\\' + qrelFile)
    results = readRes(os.getcwd() + '\\Result Files\\' + resFile)
    calc(option, relevant, results)

main()