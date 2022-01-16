/*
This program tests the icompressible instance of the MPI SUPG fluid solver; it
uses the assembly portion of the solver as implemented in mpi_insim_supg.cpp
file and the other components of the solver as implemented in
mpi_supg_solver.cpp file. It is a pure fluid 2D test case at Re=200
The final horizontal velocity profile is linear with a positive gradient.
The test runs for about 35s.
*/
#include "mpi_insim_supg.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::MPI::SUPGInsIM<2>;
extern template class Fluid::MPI::SUPGInsIM<3>;

using namespace dealii;

int main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      std::string infile("parameters.prm");
      if (argc > 1)
        {
          infile = argv[1];
        }
      Parameters::AllParameters params(infile);

      double L = 2, D = 0.4, d = 0.025, l = 0.1;
      if (params.dimension == 2)
        {
          parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
          dealii::GridGenerator::subdivided_hyper_rectangle(
            tria,
            {static_cast<unsigned int>(L / l),
             static_cast<unsigned int>(D / d)},
            Point<2>(0, 0),
            Point<2>(L, D),
            true);
          Fluid::MPI::SUPGInsIM<2> flow(tria, params);
          flow.run();
          auto solution = flow.get_current_solution();
          // Check the l1_norm of the pressure. This is supposed to come out
          // as zero since the pressure is zero everywhere.
          auto p = solution.block(1);
          double l1_norm = p.l1_norm();
          AssertThrow(l1_norm < 1e-3,
                      ExcMessage("The pressure is not zero everywhere!"));
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
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
      std::cerr << std::endl
                << std::endl
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
