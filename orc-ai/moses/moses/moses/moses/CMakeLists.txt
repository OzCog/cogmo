#install header files
INSTALL(FILES
	complexity.h
	distributed_moses.h
	neighborhood_sampling.h
	local_moses.h
	moses_main.h
	moses_params.h
	mpi_moses.h
	partial.h
	types.h
	DESTINATION "include/${PROJECT_NAME}/moses/moses"
)
