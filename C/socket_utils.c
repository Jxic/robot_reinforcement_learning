#include "socket_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include "rl_ddpg.h"
#include <assert.h>

#define PORT 6666
#define HOST "127.0.0.1"

#define DEMO_PORT 5555

#define INFO_DIM 2
#define FLAG_DIM 2

#define DEMO_INFO_DIM 1

int socket_;
struct sockaddr_in server;

int demo_socket_;
struct sockaddr_in demo_server;

int init_connection(int port) {
  socket_ = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_ == -1) {
    printf("[INIT_CONNECTION] Failed to initialize socket ...\n");
    exit(1);
  }
  
  server.sin_addr.s_addr = inet_addr(HOST);
  server.sin_family = AF_INET;
  server.sin_port = htons(port);

  if ((connect(socket_, (struct sockaddr*)&server, sizeof(server))) != 0) {
    printf("[INIT_CONNECTION] Failed to establish connection ...\n");
    exit(1);
  }
  return 1;
}

int init_demo_connection(int port) {
  demo_socket_ = socket(AF_INET, SOCK_STREAM, 0);
  if (demo_socket_ == -1) {
    printf("[INIT_DEMO_CONNECTION] Failed to initialize socket ...\n");
    exit(1);
  }

  demo_server.sin_addr.s_addr = inet_addr(HOST);
  demo_server.sin_family = AF_INET;
  demo_server.sin_port = htons(port);

  if ((connect(demo_socket_, (struct sockaddr*)&demo_server, sizeof(demo_server))) != 0) {
    printf("[INIT_CONNECTION] Failed to establish connection to %d ...\n", port);
  }
  return 1;
}

experience_buffer* build_demo_buffer(int size, int transition_dim) {
  experience_buffer* ret = init_experience_buffer(size);
  double signals[DEMO_INFO_DIM];
  double reply[transition_dim];
  signals[0] = 1;
  for (int i = 0; i < size; ++i) {
    int w = write(demo_socket_, signals, DEMO_INFO_DIM*sizeof(double));
    if (w != DEMO_INFO_DIM*sizeof(double)) {
      printf("[SEND] Failed to send full packet\n");
      exit(1);
    }
    
    int r = read(demo_socket_, reply, transition_dim*sizeof(double));
    if (r != transition_dim*sizeof(double)) {
      printf("[SEND] Failed to read full response\n");
      exit(1);
    }

    matrix_t* new_transition = new_matrix(1, transition_dim);
    for (int j = 0; j < transition_dim; ++j) new_transition->data[j] = reply[j];
    //print_matrix(new_transition, 1);
    store_experience(ret, new_transition);
  }

  return ret;
}

matrix_t* sim_send(matrix_t* t, int* flag, int state_dim, int act_dim) {
  // 0 0 initialize
  // 0 1 step
  // 1 0 reset
  // 1 1 close
  assert(t->rows==1);
  assert(t->cols==act_dim);
  
  int send_size = act_dim  + FLAG_DIM;
  int receive_size = state_dim + INFO_DIM;
  double server_reply[receive_size];
  double packet[send_size];
  packet[0] = (double)flag[0];
  packet[1] = (double)flag[1];
  free(flag);
  for (int i = 2; i < send_size; ++i) {
    packet[i] = t->data[i-2];
  }
  int w = write(socket_, packet, send_size*sizeof(double));
  if (w != send_size*sizeof(double)) {
    printf("[SEND] Failed to send full packet\n");
    exit(1);
  }
  //printf("sent %d", send_size);
  int r = read(socket_, server_reply, receive_size*sizeof(double));
  if (r != receive_size*sizeof(double)) {
    printf("[SEND] Failed to read full response\n");
    exit(1);
  }
  matrix_t* ret = new_matrix(1, receive_size);
  for (int i = 0; i < receive_size; ++i) ret->data[i] = server_reply[i];
  return ret;
}

void close_connection() {
  close(socket_);
}
