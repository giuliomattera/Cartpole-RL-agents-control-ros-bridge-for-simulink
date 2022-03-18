#include "config.cpp"
#include <math.h>
#include <iostream>

float random_number(float mean, float variance){

    float x,y,r;
    x = (float)rand()/RAND_MAX;
    y = (float)rand()/RAND_MAX;
    r = cos(2*3.14*y)*sqrt(-2*log(x));

    return mean + variance*r;
}

float frand(int xmin, int xmax){
    float r;
    r = (float)rand()/RAND_MAX;
    return xmin + r*(xmax-xmin);
}

void ini_net(int ni, int nh, int no)
{
    int i, j;
    float variance;
    inp = ni; hid = nh; out = no;
    if (ni > NUMINPUT){
        std::cout<< "The number of inputs exceded the limits";
    }
    if (no> NUMOUTPUT){
        std::cout<< "The number of outputs exceded the limits";
    }
    if (nh> NUMHIDDENNEURONS){
        std::cout<< "The number of hidden layers exceded the limits";
    }
    alpha = ALPHA0; momentum = MU0;
    variance = sqrt(1/inp);
    // Hidden layer
    for (j=0; j<hid; j++) {
        for (i=0; i<inp; i++)
            hidden_weights[j][i] = random_number(0, variance);
        bh[j] = random_number(0, variance);
    }
    // Output layer
    for (j=0; j<out; j++){
        for (i=0; i<inp; i++)
            output_weights[j][i] = random_number(0, variance);
        bo[j] = random_number(0, variance);
    }   
}

void inference() {
    int i, j;
    float sum;

    for (j=0; j<hid; j++){ 
        sum = 0;
        for (i=0; i<inp; i++)
            sum += hidden_weights[j][i]*x[i];
        h[j] = tanh(sum+bh[j]);
    }    

    for (j=0; j<hid; j++){ 
        sum = 0;
        for (i=0; i<inp; i++)
            sum += output_weights[j][i]*h[i];
        y[j] = tanh(sum+bo[j]);
    }    

}

float error_comp(int k){
    int j;
    float d, err = 0;
    for (j=0; j<out; j++){
        d = target[k][j] - y[j];
        err += d*d/2;
    }
    return err;
}

void update(int k){
    int s, i, j;
    float deltao[NUMOUTPUT];
    float deltah;
    float sum;
    float der_tanh;
    // learning backprop on output layer
    for(j=0; j<out; j++){
        der_tanh = 1 - tanh(y[j])*tanh(y[j]);
        deltao[j] = y[j]*(1-der_tanh)*(target[k][j]-y[j]);
        dbo[j] = alpha*deltao[j] + momentum*dbo[j];
        bo[j] += dbo[j];
        for (i=0; i<hid; i++)
            dwo[j][i] = alpha*deltao[j]*h[i]+momentum*dwo[j][i];
            output_weights[j][i] += dwo[j][i];
    }
    // learning backprop on hidden layer

    for(j=0; j<hid; j++){
       sum = 0;
       for (i =0 ; i < out; i++)
            sum += output_weights[i][j]*deltao[i];
        deltah = h[j]-(1-h[j])*deltao[j]*sum;
        dbh[j] = alpha*deltah + momentum*dbh[j];
        bh[j] += dbh[j];
        for (s=0; s<inp; s++)
            dwh[j][s] = alpha*deltah*x[s]+momentum*dwh[j][s];
            hidden_weights[j][s] += dwh[j][s];
    }

}

void convert_input(int k){
    int i;
    for (i=0; i<inp; i++){
        x[i] = inputs[k][i];
    }

}

float learning_step(int k){

    float err_step;
    convert_input(k);
    inference();
    err_step = error_comp(k);
    update(k);

    return err_step;

}

void learning(float eps, int maxiter){
    int k =0;
    int iter = 0;
    float gerr; //global error
    do {
        gerr = learning_step(k);
        k++;
        iter++;
        std::cout << iter;
    }
    while ((iter < maxiter) && (gerr > eps));
}

float alpha_decay(float a, float decay, int k){
    
    return alpha = a*(1/(1+decay*k));
}

