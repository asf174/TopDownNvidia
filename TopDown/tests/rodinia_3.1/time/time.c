double time() {
	struct timeval time;
  	gettimeofday(&time,NULL); // take time
	return time.tv_sec*1000.0 + time.tv_usec/1000.0;
}
