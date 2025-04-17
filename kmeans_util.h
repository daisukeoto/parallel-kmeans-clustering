#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdarg.h>
#include <ctype.h>
#include <stdint.h>
#include <math.h>

int filestats(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines);