#include <iostream>

void add(int n, float *x, float *y){

    for(int i = 0; i<n; i++){
        y[i] = x[i] + y[i];
    }
}


int main(void){

    // Declaration

    int N = 1<<20;
    float *x = new float[N];
    float *y = new float[N];

    //CPU: initialize x and y

    for(int i =0; i<N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // CPU: run on 1M elements
    add(N, x, y);

    delete [] x;
    delete [] y;

    return 0;

}