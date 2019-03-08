#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <vector>

#include "File.h"
#include "Ngram.h"
#include "VocabMap.h"
#include "Prob.h"
#include "LM.h"
#include "Vocab.h"

#ifndef MAXLEN
#define MAXLEN 256
#endif

#ifndef CANDIDATE
#define CANDIDATE 1024
#endif

#ifndef PSUDOZERO
#define PSUDOZERO -100
#endif

Vocab voc;
Vocab ZhuYin, Big5;

LogP Viterbi(Ngram& lm, VocabMap& map, VocabString* Sentence_in, unsigned count){
    // common variables
    LogP prob_matrix[MAXLEN][CANDIDATE] = {{0.0}};
    VocabIndex Vindex[MAXLEN][CANDIDATE];
    int Backtrack[MAXLEN][CANDIDATE] = {{0}};
    int Number[MAXLEN] = {0};
    // initial (t = 0)
    // Prob from Prob.h
    Prob p;
    VocabIndex vid;
    // Vocab_None from Vocab.h
    VocabIndex empty_context[] = {Vocab_None};
    VocabIndex big_context[] = {Vocab_None, Vocab_None};
    // VocabMapIter from VocabMap.h
    VocabMapIter iter(map, ZhuYin.getIndex(Sentence_in[0]));
    int len = 0;
    while(iter.next(vid, p)){
    	VocabIndex candindex = voc.getIndex(Big5.getWord(vid));
        // Vocab_None, Vocab_Unknown from Vocab.h
        if(candindex == Vocab_None){
            candindex = voc.getIndex(Vocab_Unknown);
        }
        // wordProb from LM.h
        LogP temp_p = lm.wordProb(candindex, empty_context);
        // LogP_Zero from Prob.h
        if(temp_p == LogP_Zero){
            prob_matrix[0][len] = PSUDOZERO;
        }
        else{
            prob_matrix[0][len] = temp_p;
        }
        Vindex[0][len] = vid;
        Backtrack[0][len] = -1;
        len += 1;
    }
    Number[0] = len;
    // Recursion ( t > 0 )
    for(int t = 1; t < count; t ++){
        VocabMapIter iter(map, ZhuYin.getIndex(Sentence_in[t]));
        len = 0;
        while(iter.next(vid, p)){
            VocabIndex candindex = voc.getIndex(Big5.getWord(vid));
            if(candindex == Vocab_None){
                candindex = voc.getIndex(Vocab_Unknown);
            }
            LogP temp_max = LogP_Zero;
            for(int i = 0; i < Number[t-1]; i ++){
            	big_context[0] = voc.getIndex(Big5.getWord(Vindex[t-1][i]));
                if(big_context[0] == Vocab_None){
                    big_context[0] = voc.getIndex(Vocab_Unknown);
                }
                LogP big_p = lm.wordProb(candindex, big_context);
                LogP uni_p = lm.wordProb(candindex, empty_context);
                if(big_p == LogP_Zero && uni_p == LogP_Zero){
                    big_p = PSUDOZERO;
                }
                big_p += prob_matrix[t-1][i];
                // update
                if(big_p > temp_max){
                    temp_max = big_p;
                    Backtrack[t][len] = i;
                }
            }
            prob_matrix[t][len] = temp_max;
            Vindex[t][len] = vid;
            len += 1;
        }
        Number[t] = len;
    }
    // find max probility
    LogP temp_max = LogP_Zero;
    int max_col = -1;
    for(int i = 0; i < Number[count - 1]; i ++){
        if(prob_matrix[count - 1][i] > temp_max){
            temp_max = prob_matrix[count - 1][i];
            max_col = i;
        }
    }
    // find bachtrack path
    // 
    VocabString back_path[MAXLEN];
    back_path[0] = "<s>";
    back_path[count - 1] = "</s>";
    for(int i = count - 1; i > 0; i --){
        back_path[i] = Big5.getWord(Vindex[i][max_col]);
        max_col = Backtrack[i][max_col];
    }
    for(int i = 0; i < count; i ++){
        if(i == count - 1){
            printf("%s%s", back_path[i], "\n");
        }
        else{
        	printf("%s%s", back_path[i], " ");
        }
    }
    return temp_max;
}


int main(int argc, char const *argv[]){ 
    Ngram lm(voc, atoi(argv[8]));	
    VocabMap map(ZhuYin, Big5);
    
    // load language model and map
    File lmFile(argv[6], "r");
    lm.read(lmFile);
    lmFile.close();

    File mapfile(argv[4], "r");
    map.read(mapfile);
    mapfile.close();

    File testfile(argv[2], "r");
    char* buffer = NULL;
    while(buffer = testfile.getline()){
    	// VocabString from Vocab.h
        VocabString Sentence_in[MAXLEN];
        unsigned int count = Vocab::parseWords(buffer, &(Sentence_in[1]), MAXLEN);
        // pattern
        Sentence_in[0] = "<s>";
        Sentence_in[count+1] = "</s>";
        count += 2;
        LogP Maxprob = Viterbi(lm, map, Sentence_in, count);
    }
    testfile.close();
    
    return 0;
}