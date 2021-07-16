#include <sys/time.h>

double g_time() {
	struct timeval time;
  	gettimeofday(&time,NULL); // take time
	return time.tv_sec + time.tv_usec/(1000.0*1000.0);
}
