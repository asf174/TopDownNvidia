#include <sys/time.h>

double time() {
	struct timeval time;
  	gettimeofday(&time,NULL); // take time
	return time.tv_sec + time.tv_usec/(1000.0*1000.0);
}
