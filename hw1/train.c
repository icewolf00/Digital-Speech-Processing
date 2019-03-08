#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hmm.h"


void forward(HMM *hmm_model, double alpha[MAX_SEQ][MAX_STATE], int seq_toint[MAX_SEQ], int seq_len){
    // forward_algorithm (ch4 p10)
    // initial part
    for(int i = 0; i < hmm_model->state_num; i++){
        alpha[0][i] = (hmm_model->initial[i]) * (hmm_model->observation[seq_toint[0]][i]);
    }
    // induction part
    for(int t = 1; t < seq_len; t ++){
        for(int j = 0; j < hmm_model->state_num; j ++){
            for(int i = 0; i < hmm_model->state_num; i ++){
                alpha[t][j] += alpha[t-1][i] * (hmm_model->transition[i][j]);
            }
            alpha[t][j] *= hmm_model->observation[seq_toint[t]][j];
        }
    }
    return;
}

void backward(HMM *hmm_model, double beta[MAX_SEQ][MAX_STATE], int seq_toint[MAX_SEQ], int seq_len){
    // backward algorithm (ch4 p14)
    // initial part
    for(int i = 0; i < hmm_model->state_num; i ++){
        beta[seq_len - 1][i] = 1.0;
    }
    // induction part
    for(int t = seq_len - 2; t >= 0; t --){
        for(int i = 0; i < hmm_model->state_num; i ++){
            for(int j = 0; j < hmm_model->state_num; j ++){
                beta[t][i] += beta[t+1][j] *(hmm_model->transition[i][j]) * (hmm_model->observation[seq_toint[t+1]][j]);
            }
        }
    }
    return;
}

void compute_gamma(HMM *hmm_model, double alpha[MAX_SEQ][MAX_STATE], double beta[MAX_SEQ][MAX_STATE], double gamma[MAX_SEQ][MAX_STATE], int seq_len){
    //compute gamma (ch4 p19)
    for(int t = 0; t < seq_len; t ++){
        double gamma_sum = 0.0;
        for(int i = 0; i < hmm_model->state_num; i ++){
            gamma[t][i] = alpha[t][i] * beta[t][i];
            gamma_sum += gamma[t][i];
        }
        for(int i = 0; i < hmm_model->state_num; i ++){
            gamma[t][i] /= gamma_sum;
        }
    }
    return;
}

void forward_backward(HMM *hmm_model, double alpha[MAX_SEQ][MAX_STATE], double beta[MAX_SEQ][MAX_STATE], double epsilon[MAX_SEQ][MAX_STATE][MAX_STATE], int seq_toint[MAX_SEQ], int seq_len){
    // forward-backward algorithm (ch4 p25)
    for(int t = 0; t < seq_len; t ++){
        double epsilon_sum = 0.0;
        for(int i = 0; i < hmm_model->state_num; i ++){
            for(int j = 0; j < hmm_model->state_num; j ++){
                epsilon[t][i][j] = alpha[t][i] * (hmm_model->transition[i][j]) * (hmm_model->observation[seq_toint[t+1]][j]) * beta[t+1][j];
                epsilon_sum += epsilon[t][i][j];
            }
        }
        for(int i = 0; i < hmm_model->state_num; i ++){
            for(int j = 0; j < hmm_model->state_num; j ++){
                epsilon[t][i][j] /= epsilon_sum;
            }
        }
    }
    return;
}

void accumulate(HMM *hmm_model, double gamma[MAX_SEQ][MAX_STATE], double epsilon[MAX_SEQ][MAX_STATE][MAX_STATE], double pi[MAX_STATE], double epsilon_all[MAX_STATE][MAX_STATE], double gamma_all_a[MAX_STATE][MAX_STATE], double gamma_obs[MAX_OBSERV][MAX_STATE], double gamma_all_b[MAX_OBSERV][MAX_STATE], int seq_toint[MAX_SEQ], int seq_len){
    // Accumulate through all samples, Not just all observations in one sample
    // to update pi
    for(int i = 0; i < hmm_model->state_num; i ++){
        pi[i] += gamma[0][i];
    }
    // to update aij
    for(int i = 0; i < hmm_model->state_num; i ++){
        for(int j = 0; j < hmm_model->state_num; j ++){
            for(int t = 0; t < (seq_len - 1); t ++){
                epsilon_all[i][j] += epsilon[t][i][j];
                gamma_all_a[i][j] += gamma[t][i];
            }
        }
    }
    // to update bi(k)
    for(int i = 0; i < hmm_model->observ_num; i ++){
        for(int j = 0; j < hmm_model->state_num; j ++){
            for(int t = 0; t < seq_len; t ++){
                if(seq_toint[t] == i){
                    gamma_obs[i][j] += gamma[t][j];
                }
                gamma_all_b[i][j] += gamma[t][j];
            }
        }
    }
    return;
}

void updateHMM(HMM *hmm_model, double pi[MAX_STATE], double epsilon_all[MAX_STATE][MAX_STATE], double gamma_all_a[MAX_STATE][MAX_STATE], double gamma_obs[MAX_OBSERV][MAX_STATE], double gamma_all_b[MAX_OBSERV][MAX_STATE], int data_len){
    // (ch4 p30)
    // update pi
    for(int i = 0; i < hmm_model->state_num; i ++){
        hmm_model->initial[i] = pi[i] / data_len;
    }
    // update aij
    for(int i = 0; i < hmm_model->state_num; i ++){
        for(int j = 0; j < hmm_model->state_num; j ++){
            hmm_model->transition[i][j] = epsilon_all[i][j] / gamma_all_a[i][j];
        }
    }
    // update bj(k)
    for(int i = 0; i < hmm_model->observ_num; i ++){
        for(int j = 0; j < hmm_model->state_num; j ++){
            hmm_model->observation[i][j] = gamma_obs[i][j] / gamma_all_b[i][j];
        }
    }
    return;
}

void train(HMM *hmm_model, char *data_name){
    FILE *train_data = open_or_die(data_name, "r");
    char seq_in[MAX_SEQ] = {'\0'};
    int seq_toint[MAX_SEQ] = {0};
    int seq_len = 0;
    int data_len = 0;
    // to update pi
    double pi[MAX_STATE] = {0.0};
    // to update aij
    double epsilon_all[MAX_STATE][MAX_STATE] = {{0.0}};
    double gamma_all_a[MAX_STATE][MAX_STATE] = {{0.0}};
    // to update bi(k)
    double gamma_obs[MAX_OBSERV][MAX_STATE] = {{0.0}}; 
    double gamma_all_b[MAX_OBSERV][MAX_STATE] = {{0.0}};
    // run when sequence in
    while(fscanf(train_data, "%s", seq_in) != EOF){
        data_len ++;
        seq_len = strlen(seq_in);
        for(int i = 0; i < seq_len ; i ++){
            seq_toint[i] = seq_in[i] - 'A';
        }
        // x axis => time t; y axis => state i
        double alpha[MAX_SEQ][MAX_STATE] = {{0.0}};
        double beta[MAX_SEQ][MAX_STATE] = {{0.0}};
        forward(hmm_model, alpha, seq_toint, seq_len);
        backward(hmm_model, beta, seq_toint, seq_len);

        double gamma[MAX_SEQ][MAX_STATE] = {{0.0}};
        compute_gamma(hmm_model, alpha, beta, gamma, seq_len);
        double epsilon[MAX_SEQ][MAX_STATE][MAX_STATE] = {{{0.0}}};
        forward_backward(hmm_model, alpha, beta, epsilon, seq_toint, seq_len);
        
        accumulate(hmm_model, gamma, epsilon, pi	, epsilon_all, gamma_all_a, gamma_obs, gamma_all_b, seq_toint, seq_len);
    }
    updateHMM(hmm_model, pi, epsilon_all, gamma_all_a, gamma_obs, gamma_all_b, data_len);
    fclose(train_data);
    return;
}

int main(int argc, char const *argv[])
{
    //get argv
    int ITERATION = atoi(argv[1]);
    char INIT_FILE_NAME[128] = {'\0'};
    strcpy(INIT_FILE_NAME, argv[2]);
    char DATA_FILE_NAME[128] = {'\0'};
    strcpy(DATA_FILE_NAME, argv[3]);
    char MODEL_NAME[128] = {'\0'};
    strcpy(MODEL_NAME, argv[4]);
    // model init
    HMM hmm_model;
    loadHMM(&hmm_model, INIT_FILE_NAME);

    // training
    for(int i = 0; i < ITERATION; i++)
    {
        train(&hmm_model, DATA_FILE_NAME);
    }
    // write model
    FILE* model_out = open_or_die(MODEL_NAME, "w");
    dumpHMM(model_out, &hmm_model);
    fclose(model_out);
    return 0;
}