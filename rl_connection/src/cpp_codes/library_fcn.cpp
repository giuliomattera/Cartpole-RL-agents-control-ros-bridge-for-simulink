#include <math.h>
#include <iostream>

// Utils functions

float random_number(float mean, float variance){

    float x,y,r;
    x = (float)rand()/RAND_MAX;
    y = (float)rand()/RAND_MAX;
    r = cos(2*3.14*y)*sqrt(-2*log(x));

    return mean + variance*r;
}

// Actor architecture

class FCNetwork{
    private:
        // Set maximum dimensions
        #define MAX_HID 100
        #define MAX_OUT 2
        #define MAX_INP 5
        #define MAX_EXE 100
    public:
        // Define parameters
        int inp = 4;
        int hid = 20;
        int out = 1;
        float alpha = 0.1;
        float momentum = 0.99;
        float decay = 1;

        static float bo[MAX_OUT];
        static float hidden_weights[MAX_HID][MAX_INP];
        static float output_weights[MAX_OUT][MAX_HID];
        static float bh[MAX_HID];
        static float h[MAX_HID];
        static float y[MAX_OUT];
        static float dwo[MAX_OUT][MAX_HID];
        static float dwh[MAX_HID][MAX_INP];
        static float dbo[MAX_OUT];
        static float dbh[MAX_HID];

        static float x[MAX_EXE][MAX_INP];
        static float target[MAX_EXE][MAX_OUT];
        static float v[MAX_INP];

        void network_ini(){
            int i, j;
            float variance_h, variance_o;
            FCNetwork();
            if (inp > MAX_INP){
                std::cout<< "The number of inputs exceded the limits";
                exit(1);
            }
            
            if (out> MAX_OUT){
                std::cout<< "The number of outputs exceded the limits";
                exit(1);
            }
            if (hid> MAX_HID){
                std::cout<< "The number of hidden layers exceded the limits";
                exit(1);
            }
            variance_h = sqrt(2/(inp+hid)); // Xavier initilization
            variance_o = sqrt(2/(out+hid));

            // Hidden layer
            for (j=0; j<hid; j++) {
                for (i=0; i<inp; i++)
                    hidden_weights[j][i] = random_number(0, variance_h);
                bh[j] = random_number(0, variance_h);
            }
            // Output layer
            for (j=0; j<out; j++){
                for (i=0; i<inp; i++)
                output_weights[j][i] = random_number(0, variance_o);
            bo[j] = random_number(0, variance_o);
            }   
        }

        void convert_input(int k){
            int i;
            for (i=0; i<inp; i++){
                v[i] = x[k][i];
            }
        }

        float * predict(int k) {
            int i, j;
            float sum;
            if (sizeof(x)/sizeof(x[0]) > inp){ //number of raws is number of samples
                std::cout << "The number of inputs are too high";
                exit(1);
            }
            
            for (j=0; j<hid; j++){ 
                sum = 0;
                for (i=0; i<inp; i++)
                    sum += hidden_weights[j][i]*v[k];
                h[j] = tanh(sum+bh[j]);
            }    

            for (j=0; j<hid; j++){ 
                sum = 0;
                for (i=0; i<inp; i++)
                    sum += output_weights[j][i]*h[i];
                y[j] = tanh(sum+bo[j]);
            }
            return y;
    }

        void update(int k){
            int s, i, j;
            float deltao[MAX_OUT];
            float deltah;
            float sum;
            float der_tanh;
            // learning backprop on output layer
            for(j=0; j<out; j++){
                der_tanh = 1 - tanh(y[j])*tanh(y[j]);
                deltao[j] = y[j]*(1-der_tanh)*(target[k][j]-y[j]);
                bo[j] = alpha*deltao[j] + momentum*dbo[j];
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
                    dwh[j][s] = alpha*deltah*x[k][s]+momentum*dwh[j][s];
                    hidden_weights[j][s] += dwh[j][s];
            }
        }

        float alpha_decay(float a, float decay, int k){
    
            return alpha = a*(1/(1+decay*k));
        }

        float learning_step(int k){

            float err_step;
            predict(k);
            //error
            update(k);

            return err_step;
        }

        float compute_error(int k){
            int j;
            float d, err = 0;
            for (j=0; j<out; j++){
                d = target[k][j] - y[j];
                err += d*d/2;
            }
            return err;
        }
};

