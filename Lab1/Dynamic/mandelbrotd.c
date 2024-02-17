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
        i1 = 2.0 * r * i1 + imag;
        r1 = r2 - i2 + real;
    }

    return MAX_ITER;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    double start_time, end_time;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    start_time = MPI_Wtime();
    int rows_per_process = HEIGHT / size;
    unsigned char *local_image = (unsigned char *)malloc(rows_per_process * WIDTH * sizeof(unsigned char));
    double comp_start_time = MPI_Wtime();
    int current_row = rank * rows_per_process;
    int completed_rows = 0;

    while (completed_rows < rows_per_process)
    {
        if (current_row >= (rank + 1) * rows_per_process)
        {
        
            break;
        }

        for (int x = 0; x < WIDTH; x++)
        {
            double real = (x - WIDTH / 2.0) * 4.0 / WIDTH;
            double imag = (current_row - HEIGHT / 2.0) * 4.0 / HEIGHT;
            int value = mandelbrot(real, imag);
            local_image[(current_row - rank * rows_per_process) * WIDTH + x] = (unsigned char)(value % 256);
        }
        current_row++;
        completed_rows++;
        int next_row = current_row % HEIGHT;
        MPI_Status status;
        int work_available = 0;
        for (int i = 0; i < size; i++)
        {
            if (i != rank)
            {
                MPI_Send(&next_row, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        for (int i = 0; i < size; i++)
        {
            if (i != rank)
            {
                int has_work;
                MPI_Recv(&has_work, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
                if (has_work)
                {
                    work_available = 1;
                    break;
                }
            }
        }

        if (!work_available)
        {
            break;
        }
    }

    double comp_end_time = MPI_Wtime();
    double comp_time = comp_end_time - comp_start_time;
    double comm_start_time = MPI_Wtime();
    unsigned char *global_image = NULL;
    if (rank == 0)
    {
        global_image = (unsigned char *)malloc(HEIGHT * WIDTH * sizeof(unsigned char));
    }

    MPI_Gather(local_image, rows_per_process * WIDTH, MPI_UNSIGNED_CHAR, global_image, rows_per_process * WIDTH, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    double comm_end_time = MPI_Wtime();
    double comm_time = comm_end_time - comm_start_time;
    double comm_comp_ratio = comm_time / comp_time;
    end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    double total_comm_comp_ratio = 0.0;
    MPI_Reduce(&comm_comp_ratio, &total_comm_comp_ratio, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        total_comm_comp_ratio /= size; 
    }
    if (rank == 0)
    {
        printf("The total com/comp ratio is : %.2f\n", total_comm_comp_ratio);
        printf("The total execution time: %lf seconds\n", elapsed_time);
        printf("The number of processes used is %d\n", size);

        FILE *fp = fopen("mandelbrot.ppm", "wb");
        if (fp == NULL)
        {
            fprintf(stderr, "Error: Unable to open the file for writing.\n");
            MPI_Finalize();
            return 1;
        }

        fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);

        for (int i = 0; i < HEIGHT; i++)
        {
            for (int j = 0; j < WIDTH; j++)
            {
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
