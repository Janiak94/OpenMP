/* TODO
 * Must refill buffer from time to time
 * Make file reading thread safe
 * Optimize calculations
*/

#include <omp.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define POINTS_PER_BUFFER 100000
#define MAX_DIST 34.64
#define OUT_BUFFER_SIZE 3465
#define MAX_NUM_OF_POINTS 2147483648

typedef struct{
	short x, y, z;
}Point;

//converts a row from the input file to a Point
static inline void str2point(char * restrict str, Point * restrict p){
	short d1,d2,d3;
	d1 = 1e4*(str[1] - '0');
	d1 += 1e3*(str[2] - '0');
	d1 += 1e2*(str[4] - '0');
	d1 += 1e1*(str[5] - '0');
	d1 += str[6] - '0';
	
	d2 = 1e4*(str[9] - '0');
	d2 += 1e3*(str[10] - '0');
	d2 += 1e2*(str[12] - '0');
	d2 += 1e1*(str[13] - '0');
	d2 += str[14] - '0';
	
	d3 = 1e4*(str[17] - '0');
	d3 += 1e3*(str[18] - '0');
	d3 += 1e2*(str[20] - '0');
	d3 += 1e1*(str[21] - '0');
	d3 += str[22] - '0';
	
	p->x = (str[0] == '+' ? d1 : -d1);
	p->y = (str[8] == '+' ? d2 : -d2);
	p->z = (str[16] == '+' ? d3 : -d3);
}

//converts float to string for output
static inline void f2str(char *str, float f){
	sprintf(str, f < 10 ? "0%.2f":"%.2f", f+0.005);
}

static inline size_t i2str(char *str, int i){
	if(i < 10){
		str[0] = i + '0';
		str[1] = '\n';
		return 2;
	}else if(i < 100){
		str[0] = i/10 + '0';
		str[1] = i%10 + '0';
		str[2] = '\n';
		return 3;
	}else{
		size_t length = 0;
		for(int j = 1; j < 1e10 && i/j != 0; j*=10){
			++length;
		}
		for(int j = 0; j < length; ++j){
			str[length-j-1] = i%10 + '0';
			i/=10;
		}
		str[length] = '\n';
		return length+1;
	}
	return 0;
}

static inline int read_points(
			FILE * restrict input_file,
			Point start_points[],
			Point end_points[],
			size_t * restrict start_buffer_size,
			size_t * restrict end_buffer_size
) {
	size_t static bytes_read,  position_in_file = 0;
	char static file_content[24*POINTS_PER_BUFFER];
	//first check if we need to read to start_points
	if(position_in_file == 0 || feof(input_file)){
		#pragma omp parallel
		{
			#pragma omp single
			{
				fseek(input_file, position_in_file, SEEK_SET);
				bytes_read = fread(file_content, 1, 24*POINTS_PER_BUFFER, input_file);
				position_in_file += bytes_read;
			}
			//read same content to start and end since this is a new triangular block
			#pragma omp for
			for(size_t current_byte = 0; current_byte < bytes_read; current_byte+=24){
				str2point(&file_content[current_byte], &start_points[current_byte/24]);
				str2point(&file_content[current_byte], &end_points[current_byte/24]);
			}
			#pragma omp single
			{
				*start_buffer_size = bytes_read/24;
				*end_buffer_size = bytes_read/24;
			}
			// return value should reflect if block or triangular or if done
		}
		return feof(input_file) ? 2:1;
	}
	#pragma omp parallel
	{
		#pragma omp single
		bytes_read = fread(file_content, 1, 24*POINTS_PER_BUFFER, input_file);
		#pragma omp for
		for(size_t current_byte = 0; current_byte < bytes_read; current_byte+=24){
			str2point(&file_content[current_byte], &end_points[current_byte/24]);
		}
		#pragma omp single
		*end_buffer_size = bytes_read/24;
	}
	return 0;
}

static inline short point_index(Point p1, Point p2){ //use intrinsics (SIMD)
	return (short)(sqrtf(
			(p1.x-p2.x)*(p1.x-p2.x) +
			(p1.y-p2.y)*(p1.y-p2.y) +
			(p1.z-p2.z)*(p1.z-p2.z)
		)/10.0);
}

static inline void calc_block(
		Point start_points[],
		Point end_points[],
		size_t start_length,
		size_t end_length,
		unsigned int output[]
) {
	#pragma omp parallel for collapse(2) reduction(+:output[:OUT_BUFFER_SIZE])
	for(int i = 0; i < start_length; ++i) {
		for(int j = 0; j < end_length; ++j) {
			++output[point_index(start_points[i], end_points[j])];
		}
	}
}

static inline void calc_triangle(
		Point start_points[],
		Point end_points[],
		size_t length,
		unsigned int output[]
) {
	#pragma omp parallel for reduction(+:output[:OUT_BUFFER_SIZE])
	for(int i = 0; i < length; ++i) {
		for(int j = i+1; j < length; ++j) {
			++output[point_index(start_points[i], end_points[j])]; //use different buffers to increase locality
		}
	}
}


int main(int argc, char *argv[]){
	if(argc != 2){
		exit(-1);
	}
	if(argv[1][0] != '-' || argv[1][1] != 't'){
		exit(-2);
	}
	size_t num_of_threads = strtol(&argv[1][2], NULL, 10);
	omp_set_num_threads(num_of_threads);
	FILE *fp;
	fp = fopen("input_files/cell_e5","r");
//	fp = fopen("cells","r");
	Point *start_points = (Point*) malloc(POINTS_PER_BUFFER*sizeof(Point));
	Point *end_points = (Point*) malloc(POINTS_PER_BUFFER*sizeof(Point));
	unsigned int output_occurance[OUT_BUFFER_SIZE];
	memset(output_occurance,0,sizeof(int)*OUT_BUFFER_SIZE);
	size_t start_length, end_length;
	int return_value;
	do{ //TODO: check the return values, that they are correct
		return_value = read_points(fp, start_points, end_points, &start_length, &end_length);
		if(return_value == 0){
			calc_block(start_points, end_points, start_length, end_length, output_occurance);
		}else if(return_value == 1){
			calc_triangle(start_points, end_points, start_length, output_occurance);
		}
	}while(return_value != 2);
	char out_string[20*OUT_BUFFER_SIZE];
	char temp[20];
	int size = 0, i2str_count = 0;
	for(int i = 0; i < 3465; ++i){
		if(output_occurance[i]){
			f2str(temp, MAX_DIST*i/(3465.0));
			temp[5] = ' ';
			i2str_count = i2str(&temp[6], output_occurance[i]);
			memcpy(&out_string[size], temp, 6+i2str_count);
			size += 6 + i2str_count;
		}
	}
	printf(out_string);

	fclose(fp);
	free(start_points);
	free(end_points);
	return 0;
}
