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

int socket_;
struct sockaddr_in server;

int init_connection() {
  socket_ = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_ == -1) {
    printf("[INIT_CONNECTION] Failed to initialize socket ...\n");
    exit(1);
  }
  
  server.sin_addr.s_addr = inet_addr(HOST);
  server.sin_family = AF_INET;
  server.sin_port = htons(PORT);

  if ((connect(socket_, (struct sockaddr*)&server, sizeof(server))) != 0) {
    printf("[INIT_CONNECTION] Failed to establish connection ...\n");
    exit(1);
  }
  return 1;
}

matrix_t* sim_send(matrix_t* t, int* flag) {
  // 0 0 initialize
  // 0 1 step
  // 1 0 reset
  // 1 1 close
  assert(t->rows==1);
  assert(t->cols==ACTION_DIM);
  
  int send_size = ACTION_DIM  + FLAG_DIM;
  int receive_size = STATE_DIM + INFO_DIM;
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
