#install header files
INSTALL(FILES
        hill-climbing.h
        optimization.h
        star-anneal.h
        univariate.h
        particle-swarm.h
	DESTINATION "include/opencog/${PROJECT_NAME}/moses/optimization"
)
