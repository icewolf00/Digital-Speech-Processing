#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hmm.h"

double viterbi(HMM *hmm_model, int seq_toint[MAX_SEQ], int seq_len){
    // viterbi algorithm (ch4 p20, p23)
    double delta[MAX_SEQ][MAX_STATE] = {{0.0}};
    // initial part
    for(int i = 0; i < hmm_model->state_num; i ++){
        delta[0][i] = (hmm_model->initial[i]) * (hmm_model->observation[seq_toint[0]][i]);
    }
    // induction part / recursion part
    for(int t = 0; t < seq_len - 1; t ++){
        for(int j = 0; j < hmm_model->state_num; j ++){
            double max_value = 0.0;
            for(int i = 0; i < hmm_model->state_num; i ++){
                double temp_value = 0;
                temp_value = delta[t][i] * (hmm_model->transition[i][j]);
                if(temp_value > max_value){
                    max_value = temp_value;
                }
            }
            delta[t+1][j] = max_value * (hmm_model->observation[seq_toint[t+1]][j]);
        }
    }
    // termination
    double max_prob = 0.0;
    for(int i = 0; i < hmm_model->state_num; i ++){
        if(delta[seq_len-1][i] > max_prob){
            max_prob = delta[seq_len-1][i];
        }
    }
    return max_prob;
}


int main(int argc, char const *argv[])
{
    char MODEL_LIST_NAME[128] = {'\0'};
    strcpy(MODEL_LIST_NAME, argv[1]);
    char DATA_FILE_NAME[128] = {'\0'};
    strcpy(DATA_FILE_NAME, argv[2]);
    char RESULT_FILE_NAME[128] = {'\0'};
    strcpy(RESULT_FILE_NAME, argv[3]);
    HMM hmm_models[5];
    load_models(MODEL_LIST_NAME, hmm_models, 5);
    
    FILE *DATA = open_or_die(DATA_FILE_NAME, "r");
    FILE *RESULT = open_or_die(RESULT_FILE_NAME, "w");
    char seq_in[MAX_SEQ] = {'\0'};
    int seq_toint[MAX_SEQ] = {0};
    int seq_len = 0;
    
    while(fscanf(DATA, "%s", seq_in) != EOF){
        seq_len = strlen(seq_in);
        for(int i = 0; i < seq_len ; i ++){
            seq_toint[i] = seq_in[i] - 'A';
        }
        double max_prob = 0.0;
        int max_model = -1; 
        for(int i = 0; i < 5; i++){
            double temp_prob = 0.0;
            temp_prob = viterbi(&hmm_models[i], seq_toint, seq_len);
            if(temp_prob > max_prob){
                max_prob = temp_prob;
                max_model = i;
            }
        }
        fprintf(RESULT, "%s %e\n", hmm_models[max_model].model_name, max_prob);
    }
    fclose(DATA);
    fclose(RESULT);
    return 0;
}