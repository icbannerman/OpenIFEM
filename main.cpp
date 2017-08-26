#include "linearElasticSolver.h"
#include "linearMaterial.h"

#include <iostream>

int main(int argc, char* argv[])
{
  try
  {
    using namespace dealii;
    using namespace IFEM;

    LinearElasticSolver<2> solver;
    if (argc > 1)
    {
      solver.readMesh(string(argv[1]));
    }
    else
    {
      solver.generateMesh();
    }
    LinearMaterial<2> steel(1., 1.);
    solver.setMaterial(steel);
    solver.readBC();
    solver.setup();
    solver.assemble();
    solver.applyBC();
    solver.output(0);
    solver.solve();
    solver.output(1);
  }
  catch (std::exception& exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}
