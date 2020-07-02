float A[1000][1000];
float a;
int main() {
#pragma scop
  a = 0.0;
  // for(int i = 0; i < 10; i++){
  for (int i = 200; i > 100; i--) {
    for (int k = 200; k > 100; k--) {
      // for (int k = 101; k <= 200; k++){
      a = a + 5.0;
      A[k][i] = 2.0f;
    }
    for (int k = 101; k <= 200; k++) {
      a = a + 5.0;
      A[i][k] = 2.0f;
    }
  }

#pragma endscop
  return a;
}
