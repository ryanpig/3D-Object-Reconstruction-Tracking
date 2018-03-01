#include <cstdlib>
#include <string>

#include "utilities/General.h"
#include "VoxelReconstruction.h"

using namespace nl_uu_science_gmt;


int main(
		int argc, char** argv)
{
	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	vr.run(argc, argv);

	return EXIT_SUCCESS;
}
