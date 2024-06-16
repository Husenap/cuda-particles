#include <cassert>
#include <iostream>
#include <vector>

#include <glad/glad.h>

#define GLFW_INCLUDE_NONE

#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include <libtetris.h>

constexpr int BOARD_WIDTH  = 10;
constexpr int BOARD_HEIGHT = 20;

constexpr int VIEWPORT_WIDTH  = 600;
constexpr int VIEWPORT_HEIGHT = VIEWPORT_WIDTH * BOARD_HEIGHT / BOARD_WIDTH;
constexpr int NUM_PARTICLES   = 1 << 20;

using board_state_t = int8_t[BOARD_HEIGHT][BOARD_WIDTH];

struct Particle {
  float2 pos;
  float2 vel;
  float3 color;
};

__global__ void create_particles(Particle* particles) {
  const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < NUM_PARTICLES) {
    curandState localState;
    curand_init(1337, idx, 0, &localState);
    Particle& p = particles[idx];
    p.pos.x     = curand_uniform(&localState);
    p.pos.y     = curand_uniform(&localState) * 2.0f;
    p.vel.x     = (curand_uniform(&localState) * 2.0f - 1.0f) * 0.01f;
    p.vel.y     = (curand_uniform(&localState) * 2.0f - 1.0f) * 0.01f;
  }
}

__global__ void update_particles(Particle* particles, board_state_t* board_state, float2* block_positions, int num_block_positions, float time, float delta_time) {
  static const float3 colors[10] = {
      {0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 1.0f},
      {1.0f, 0.5f, 0.0f},
      {1.0f, 1.0f, 0.0f},
      {1.0f, 0.0f, 0.0f},
      {1.0f, 0.0f, 1.0f},
      {0.0f, 0.0f, 1.0f},
      {0.2f, 1.0f, 0.2f},
      {0.0f, 0.0f, 0.0f},
      {0.5f, 0.5f, 0.5f},
  };

  const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < NUM_PARTICLES) {
    Particle& p = particles[idx];

    const int initial_block_x = max(min(static_cast<int>(p.pos.x * BOARD_WIDTH), BOARD_WIDTH - 1), 0);
    const int initial_block_y = max(min(static_cast<int>(p.pos.y * 0.5f * BOARD_HEIGHT), BOARD_HEIGHT - 1), 0);

    const float2 target{.x = (static_cast<float>(initial_block_x) + 0.5f) / BOARD_WIDTH,
                        .y = (static_cast<float>(initial_block_y) + 0.5f) / BOARD_HEIGHT * 2.0f};

    const float2 acc{.x = target.x - p.pos.x, .y = target.y - p.pos.y};

    const float dt  = delta_time;
    const float dt2 = dt * dt;

    p.vel.x += dt * acc.x;
    p.vel.y += dt * acc.y;

    p.pos.x += dt * p.vel.x + dt2 * acc.x * 0.5f;
    p.pos.y += dt * p.vel.y + dt2 * acc.y * 0.5f;

    const int block_x = max(min(static_cast<int>(p.pos.x * BOARD_WIDTH), BOARD_WIDTH - 1), 0);
    const int block_y = max(min(static_cast<int>(p.pos.y * 0.5f * BOARD_HEIGHT), BOARD_HEIGHT - 1), 0);

    const auto alpha = delta_time * 20.0f;
    const auto color = colors[(*board_state)[block_y][block_x] + 1];
    p.color.x        = (1.0f - alpha) * p.color.x + alpha * color.x;
    p.color.y        = (1.0f - alpha) * p.color.y + alpha * color.y;
    p.color.z        = (1.0f - alpha) * p.color.z + alpha * color.z;
  }
}

const char* vertexShaderSource   = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aColor;
out vec3 iColor;
void main() {
    gl_Position = vec4(aPos.x*2.0f-1.0f, 1.0f-aPos.y, 0.0, 1.0);
    iColor = aColor;
}
)";
const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in vec3 iColor;
void main() {
    FragColor = vec4(iColor, 0.1);
}
)";

static tetris_inputs_t key_input{};

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  switch (key) {
  case GLFW_KEY_DOWN:
    key_input.soft_drop = action != GLFW_RELEASE;
    break;
  case GLFW_KEY_UP:
    key_input.rotate_cw = action != GLFW_RELEASE;
    break;
  case GLFW_KEY_LEFT:
    key_input.left = action != GLFW_RELEASE;
    break;
  case GLFW_KEY_RIGHT:
    key_input.right = action != GLFW_RELEASE;
    break;
  case GLFW_KEY_SPACE:
    key_input.hard_drop = action != GLFW_RELEASE;
    break;
  case GLFW_KEY_LEFT_SHIFT:
    key_input.hold = action != GLFW_RELEASE;
    break;
  case GLFW_KEY_LEFT_CONTROL:
    key_input.rotate_ccw = action != GLFW_RELEASE;
    break;
  default:
    break;
  }
}

int main() {
  GLFWwindow* window;
  { // Initialize GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // Create window
    window = glfwCreateWindow(VIEWPORT_WIDTH, VIEWPORT_HEIGHT, "CUDA Particles", nullptr, nullptr);
    glfwSetKeyCallback(window, key_callback);
    glfwMakeContextCurrent(window);
  }

  { // Initialize OpenGL
    assert(gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)));
    glViewport(0, 0, VIEWPORT_WIDTH, VIEWPORT_HEIGHT);
    glfwSwapInterval(0);
  }

  cudaSetDevice(0);

  GLuint                 particlesVBO, particlesVAO;
  cudaGraphicsResource_t particlesVBO_CUDA;
  { // Create particles buffer
    glGenBuffers(1, &particlesVBO);
    glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
    glBufferData(GL_ARRAY_BUFFER, NUM_PARTICLES * sizeof(Particle), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&particlesVBO_CUDA, particlesVBO, cudaGraphicsMapFlagsWriteDiscard);

    glGenVertexArrays(1, &particlesVAO);
    glBindVertexArray(particlesVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (GLvoid*)nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (GLvoid*)offsetof(Particle, color));
    glEnableVertexAttribArray(1);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glBindVertexArray(0);
  }

  GLuint program;
  { // Create particle shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);
    int  success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
      std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
      std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
  }

  { // Initialize particles
    Particle* particles;
    cudaGraphicsMapResources(1, &particlesVBO_CUDA);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&particles), &num_bytes, particlesVBO_CUDA);
    const dim3 dim_block = 1024;
    const dim3 dim_grid  = (NUM_PARTICLES + dim_block.x - 1) / dim_block.x;
    create_particles<<<dim_grid, dim_block>>>(particles);
    cudaGraphicsUnmapResources(1, &particlesVBO_CUDA);
  }

  tetris_t*           tetris_game = create_game();
  board_state_t       board_state;
  board_state_t*      d_board_state;
  std::vector<float2> block_positions;
  float2*             d_block_positions;
  block_positions.reserve(BOARD_WIDTH * BOARD_HEIGHT);
  init(tetris_game, BOARD_WIDTH, BOARD_HEIGHT, 1000000, 166667, 33000);
  cudaMalloc(reinterpret_cast<void**>(&d_board_state), sizeof(board_state_t));
  cudaMalloc(reinterpret_cast<void**>(&d_block_positions), sizeof(float2) * BOARD_WIDTH * BOARD_HEIGHT);
  const auto update_board_state = [&]() {
    block_positions.clear();
    for (coord_t y = 0; y < BOARD_HEIGHT; ++y) {
      for (coord_t x = 0; x < BOARD_WIDTH; ++x) {
        board_state[y][x] = read_game(tetris_game, x, y);
        if (board_state[y][x] >= 0) {
          block_positions.push_back({.x = static_cast<float>(x), .y = static_cast<float>(y)});
        }
      }
    }
    cudaMemcpy(d_board_state, board_state, sizeof(board_state_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_positions, block_positions.data(), sizeof(float2) * BOARD_WIDTH * BOARD_HEIGHT, cudaMemcpyHostToDevice);
  };
  update_board_state();

  double time           = glfwGetTime();
  int    frame          = 0;
  double delta_time_acc = 0.0;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    const auto current_time = glfwGetTime();
    const auto frame_time   = current_time - time;
    time                    = current_time;

    delta_time_acc += frame_time;
    ++frame;
    if (delta_time_acc >= 1.0) {
      const auto delta_time_per_frame = delta_time_acc / frame;
      std::cout << "frame time: " << delta_time_per_frame * 1'000.0 << "ms" << std::endl;
      std::cout << "fps: " << (1.0 / delta_time_per_frame) << std::endl;
      delta_time_acc -= 1.0;
      frame = 0;
    }

    const auto delta_time = std::min(frame_time, 0.02);

    if (tick(tetris_game, tetris_params_t{.inputs = key_input, .delta_time = static_cast<time_us_t>(delta_time * 1'000'000.0)})) {
      update_board_state();
    }

    { // Update particles
      Particle* particles;
      cudaGraphicsMapResources(1, &particlesVBO_CUDA);
      size_t num_bytes;
      cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&particles), &num_bytes, particlesVBO_CUDA);
      const dim3 dim_block = 1024;
      const dim3 dim_grid  = (NUM_PARTICLES + dim_block.x - 1) / dim_block.x;
      update_particles<<<dim_grid, dim_block>>>(particles, d_board_state, d_block_positions, block_positions.size(), static_cast<float>(time), static_cast<float>(delta_time));
      cudaGraphicsUnmapResources(1, &particlesVBO_CUDA);
    }

    glClearColor(0.05f, 0.06f, 0.07f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(program);
    glBindVertexArray(particlesVAO);
    glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

    glfwSwapBuffers(window);
  }

  destroy_game(tetris_game);

  { // Destroy particles
    cudaGraphicsUnregisterResource(particlesVBO_CUDA);
    glDeleteBuffers(1, &particlesVBO);
  }

  { // Destroy GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
  }

  return 0;
}
