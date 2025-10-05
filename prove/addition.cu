#include <iostream>

void add(int n, float *x, float *y, float*f){

    for(int i = 0; i<n; i++){
        f[i] = x[i] + y[i];
    }
}


int main(void){

    // Declaration

    int N = 1<<20;
    float *x = new float[N];
    float *y = new float[N];
    float *f = new float[N]

    //CPU: initialize x and y

    for(int i =0; i<N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // CPU: run on 1M elements
    add(N, x, y, f);


    delete [] x;
    delete [] y;
    delete [] f;

    return 0;

}