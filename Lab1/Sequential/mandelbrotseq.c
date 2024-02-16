#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 1000

#define HEIGHT 1000

#define MAX_ITER 1000


int mandelbrot(double real, double imag)
{
    int n;
    double r1 = 0.0;
    double i1 = 0.0;

    for (n = 0; n < MAX_ITER; n++)
    {
        double r2 = r1 * r1;
        double i2 = i1 * i1;
        if (r2 + i2 > 4.0)
        {
            return n;
        }
        i1 = 2.0 * r1 * i1 + imag;
        r1 = r2 - i2 + real;
    }
    return MAX_ITER;
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    double start_time, end_time;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    start_time = MPI_Wtime();
    int rows_per_process = HEIGHT / size;
    unsigned char *local_image = (unsigned char *)malloc(rows_per_process * WIDTH * sizeof(unsigned char));
    double comp_start_time = MPI_Wtime();
    for (int y = rank * rows_per_process; y < (rank + 1) * rows_per_process; y++){
        for (int x = 0; x < WIDTH; x++)
        {
            double real = (x - WIDTH / 2.0) * 4.0 / WIDTH;
            double imag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;
            int value = mandelbrot(real, imag);
            local_image[(y - rank * rows_per_process) * WIDTH + x] = (unsigned char)(value % 256);
        }
    }
    double comp_end_time = MPI_Wtime();
    double comp_time = comp_end_time - comp_start_time;
    double comm_start_time = MPI_Wtime();
    unsigned char *global_image = NULL;
    if (rank == 0){
        global_image = (unsigned char *)malloc(HEIGHT * WIDTH * sizeof(unsigned char));
    }

    MPI_Gather(local_image, rows_per_process * WIDTH, MPI_UNSIGNED_CHAR, global_image, rows_per_process * WIDTH, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    double comm_end_time = MPI_Wtime();
    double comm_time = comm_end_time - comm_start_time;
    double comm_comp_ratio = comm_time / comp_time;
    end_time = MPI_Wtime();
    double total_execution_time = end_time - start_time;
    double total_comm_comp_ratio = 0.0;
    MPI_Reduce(&comm_comp_ratio, &total_comm_comp_ratio, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0){
        total_comm_comp_ratio /= size; 
    }
    if (rank == 0){
        printf("The com/comp ratio is : %.2f\n", total_comm_comp_ratio);
        printf("Total execution time: %lf seconds\n", total_execution_time);
        printf("the number of processes is %d\n", size);

        FILE *fp = fopen("mandelbrot.ppm", "wb");
        if (fp == NULL){
            fprintf(stderr, "Error: Unable to open the file for writing.\n");
            MPI_Finalize();
            return 1;
        }
        fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
        for (int i = 0; i < HEIGHT; i++){
            for (int j = 0; j < WIDTH; j++){
                unsigned char pixel_value = global_image[i * WIDTH + j];
                fputc(pixel_value, fp);
                fputc(pixel_value, fp); 
                fputc(pixel_value, fp);
            }
        }
        fclose(fp);

        free(global_image);
    }
    free(local_image);

    MPI_Finalize();

    return 0;
}
