#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <vector_types.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>


__global__ void gaussianKernel(uchar4* blurredImage, uchar4* originalImage, float* filter, int height, int width, int filterWidth);

float* createFilter(int width) {
    const float sigma = 2.f;                         // Standard deviation of the Gaussian distribution.
    const int       half = width / 2;
    float           sum = 0.f;

    // Create convolution matrix
    float* res = (float*)malloc(width * width * sizeof(float));

    // Calculate filter sum first
    for (int r = -half; r <= half; ++r) {
        for (int c = -half; c <= half; ++c) {
            // e (natural logarithm base) to the power x, where x is what's in the brackets
            float weight = expf(-static_cast<float>(c * c + r * r) / (2.f * sigma * sigma));
            int idx = (r + half) * width + c + half;

            res[idx] = weight;
            sum += weight;
        }
    }

    // Normalize weight: sum of weights must equal 1
    float normal = 1.f / sum;

    for (int r = -half; r <= half; ++r) {
        for (int c = -half; c <= half; ++c) {
            int idx = (r + half) * width + c + half;
            res[idx] *= normal;
        }
    }
    return res;
}

// Main entry into the application
int main(int argc, char** argv) {

    //CREACIÓN DE VARIABLES NECESARIAS
    char* imagePath;
    char* outputPath; 
    int height, width, bpp, channels = 4;
    uchar4* originalImage, * blurredImage; 
    int filterWidth = 81; //Anchura de filtro: matriz de desenfoque de (9x9)
    float* filter = createFilter(filterWidth); //Llamo a la función que genera la matriz del filtro de desenfoque dadas las dimensiones de esa matriz REFERENCIA 5

    if (argc > 2) {
        imagePath = argv[1];    //Directorio de imagen de entrada
        outputPath = argv[2];   //Directorio de imagen de salida
    } else {
        printf("Please provide input and output image files as arguments to this application.");
        exit(1);
    }

    //Leer la imagen
    uint8_t* rgb_image = stbi_load(imagePath, &width, &height, &bpp, channels); //Función definida en la libreria stb_image.h REFERENCIA 6

    if (rgb_image == NULL) printf("Could not load image file: %s\n", imagePath);

    //Allocate and copy
    originalImage = (uchar4*)malloc(width * height * sizeof(uchar4));    //Inicializo la imagen original como la tupla de 4 elementos
    blurredImage = (uchar4*)malloc(width * height * sizeof(uchar4));     //Inicializo la imagen de salida como la tupla de 4 elementos
    printf("Width:%d, Height:%d Size(in Bytes):%lu\n", width, height, width * height * bpp * channels);
    
    for (int i = 0; i < width * height * channels; i++) {
        int mod = i % channels;
        switch (mod) {
        case 0:
            originalImage[i / channels].x = rgb_image[i];
            break;
        case 1:
            originalImage[i / channels].y = rgb_image[i];
            break;
        case 2:
            originalImage[i / channels].z = rgb_image[i];
            break;
        case 3:
            originalImage[i / channels].w = rgb_image[i];
            break;
        }
    }

    //Duplico las matrices para pasarselas a la GPU
    uchar4* originalImageGPU;
    uchar4* blurredImageGPU;
    float* filterGPU;

    //ALOCATO MEMORIA EN LA GPU para las matrices de GPU
    cudaMalloc(&originalImageGPU,(width * height * sizeof(uchar4)));
    cudaMalloc(&blurredImageGPU, (width * height * sizeof(uchar4)));
    cudaMalloc(&filterGPU, (filterWidth*filterWidth*sizeof(float)));

    //COPIO MATRICES A LA GPU (matrixGPU, matrixCPU, size, cudaMemcpyHostToDevice)
    cudaMemcpy(originalImageGPU, originalImage, (width * height * sizeof(uchar4)), cudaMemcpyHostToDevice);
    cudaMemcpy(filterGPU, filter, filterWidth*filterWidth*sizeof(float), cudaMemcpyHostToDevice);
   
    int threads = 32;
    int x, y;
    if (width % threads != 0) {
        x = width / threads + 1;
    }
    else { 
        x = width / threads;
    }
    if (height % threads != 0) {
        y = height / threads + 1;

    } else {
        y = height / threads;
    }

    //Parámetros cuda:
    dim3 blocks(x, y);           //nums bloques en x e y TODO PARA QUE SEA GENERICO ARAMETRIZAR E CUATRO EN FUNCIÓ DEL TAMAÑO DE IMAGEN
    dim3 threads_per_block(threads, threads); //Threads por bloque en x e y

   //Variables para medir el tiempo de ejecución del kernel
   float time;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);


   cudaEventRecord(start, 0);       //Realizo medición inicial
   gaussianKernel<<<blocks,threads_per_block>>>(blurredImageGPU, originalImageGPU, filterGPU, height, width, filterWidth); //Lanzo función de kernel

   cudaThreadSynchronize(); //Espero a que el kernel acabe

   cudaEventRecord(stop, 0);       //Realizo medición final
   cudaEventSynchronize(stop);

   //Leo los resultados y los copio en blurredImage
   cudaMemcpy(blurredImage, blurredImageGPU, (width * height * sizeof(uchar4)), cudaMemcpyDeviceToHost);


    //Formateo la salida en una matriz no vectorial
    for (int i = 0; i < width * height; i++) {
        rgb_image[i * channels] = blurredImage[i].x;
        rgb_image[(i * channels) + 1] = blurredImage[i].y;
        rgb_image[(i * channels) + 2] = blurredImage[i].z;
        rgb_image[(i * channels) + 3] = blurredImage[i].w;
    }

    //Llamo a la función que printea la matriz no vectorial
    stbi_write_jpg(outputPath, width, height, 4, blurredImage, 100);     //Función definida en stb_image_write.h
    printf("Done!\n");

    //Libero el espacio en GPU
    cudaFree(originalImageGPU);
    cudaFree(blurredImageGPU);
    cudaFree(filterGPU);

    cudaEventElapsedTime(&time, start, stop);
    printf("Time for the kernel: %f ms\n", time);

    return 0;
}

__global__ void gaussianKernel(uchar4* blurredImage, uchar4* originalImage, float* filter, int height, int width, int filterWidth) {
    //Identificar al thread
    int col= (blockDim.x * blockIdx.x)+threadIdx.x; //identifico x: ( (ancho/4)* identificador del bloque coordenada x) + coordenada x dentro del bloque
    int row= (blockDim.y * blockIdx.y)+threadIdx.y; //identifico y(altura): ( 2 * identificador del bloque coordenada y ) + coordenada y dentro del bloque

    const int numPixels = width * height;
    const int half = filterWidth / 2;
    float blur_red = 0.0;
    float blur_green = 0.0;
    float blur_blue = 0.0;

   if ((col + (width * row)) < numPixels ) {
      
       //Elementos de salida
       unsigned char redBlurred = 0;
       unsigned char greenBlurred = 0;
       unsigned char blueBlurred = 0;

       //Average pixel color summing up adjacent pixels.
       for (int i = -half; i <= half; ++i) {
           for (int j = -half; j <= half; ++j) {
               // Clamp filter to the image border (gestión de los bordes de la imagen)
               int h = min(max(row + i, 0), height-1); //row= row en la que estoy
               int w = min(max(col + j, 0), width-1);  //col= columna en la que estoy

               int idx_matrix = (w + (width)*h); //Identifico el pixel de la matriz actual

               //Identifico las tres componentes del pixel a computar(casteo de char a float)
               unsigned char pixel_red = originalImage[idx_matrix].x;
               unsigned char pixel_green = originalImage[idx_matrix].y;
               unsigned char pixel_blue = originalImage[idx_matrix].z;

               int idx_filter = (i + half) * filterWidth + j + half;  //Id del filtro

               float weight = filter[idx_filter];                     //Saco el peso por el que operar para desenfocar  
              
               blur_red += pixel_red * weight;                        //Calculo la componente roja del pixel desenfocada
               blur_green += pixel_green * weight;                    //Calculo la componente verde del pixel desenfocada
               blur_blue += pixel_blue * weight;                      //Calculo la componente azul del pixel desenfocada
               
           }
            blurredImage[col + width * row].x = blur_red;
            blurredImage[col + width * row].y = blur_green;
            blurredImage[col + width * row].z = blur_blue;
       }

        //Paso de float a Char
        redBlurred = blur_red;
        greenBlurred = blur_green;
        blueBlurred = blur_blue;

        blurredImage[col + width * row] = { redBlurred, greenBlurred, blueBlurred, 255 };
      
   }
}