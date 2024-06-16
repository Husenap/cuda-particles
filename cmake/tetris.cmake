message("-- External Project: libtetris")

add_library(tetris STATIC
        libtetris/src/bag.c
        libtetris/src/framebuffer.c
        libtetris/src/lcg_rand.c
        libtetris/src/libtetris.c
        libtetris/src/piece.c)
set_target_properties(tetris PROPERTIES LANGUAGE C)
target_include_directories(tetris INTERFACE libtetris/src)
