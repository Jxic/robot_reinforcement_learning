#include <vector>
#include <string>
#include "setup.hpp"
#include <iostream>
#include "tests.hpp"


using namespace std;

int _main() {
  vector<string> kns;
  kns.push_back("vector_add");
  kns.push_back("gemm");
  Config c = init_opencl(kns);
  c.show();

  cout << endl;
  cout << "Starting tests" << endl;

  kernel_tests(kns);


  return 0;
}