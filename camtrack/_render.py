#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from collections import namedtuple
from typing import List, Tuple

import numpy as np
import cv2
from transforms3d import quaternions
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL import GLUT
from OpenGL.arrays import vbo

import data3d


def _transform_track(track, point_cloud):
    point_cloud_points = point_cloud.points
    cv_to_gl_rotate = np.zeros((4, 4))
    cv_to_gl_rotate[0][0] = 1
    cv_to_gl_rotate[1][1] = -1
    cv_to_gl_rotate[2][2] = -1
    cv_to_gl_rotate[3][3] = 1

    for i in range(point_cloud_points.shape[0]):
        point_cloud_points[i] = cv_to_gl_rotate[0:3, 0:3].dot(point_cloud_points[i])
    for i in range(track.shape[0]):
        track[i] = cv_to_gl_rotate.dot(track[i]).dot(cv_to_gl_rotate)


def _projection_matrix(fov_y, aspect_ratio, near, far):
    focal_len_y = 1 / np.tan(fov_y / 2)
    z_diff = near - far
    return np.array(
        [[focal_len_y / aspect_ratio, 0, 0, 0],
         [0, focal_len_y, 0, 0],
         [0, 0, (near + far) / z_diff, -1],
         [0, 0, 2 * near * far / z_diff, 0]]
    ).transpose()


def _generate_frustrum(tracked_cam_projection):
    tracked_cam_projection_inv = np.linalg.inv(tracked_cam_projection)
    camera_space_frustrum = \
        [[-1, -1, -1, 1], [-1, -1, 1, 1],
         [1, -1, -1, 1], [1, -1, 1, 1],
         [-1, 1, -1, 1], [-1, 1, 1, 1],
         [1, 1, -1, 1], [1, 1, 1, 1],
         [-1, -1, -1, 1], [-1, 1, -1, 1],
         [1, 1, -1, 1], [-1, 1, -1, 1],
         [1, 1, -1, 1], [1, -1, -1, 1],
         [-1, -1, -1, 1], [1, -1, -1, 1],
         [-1, -1, 1, 1], [-1, 1, 1, 1],
         [1, 1, 1, 1], [-1, 1, 1, 1],
         [1, 1, 1, 1], [1, -1, 1, 1],
         [-1, -1, 1, 1], [1, -1, 1, 1]]
    global_space_frustrum = \
        [tracked_cam_projection_inv.dot(p) for p in camera_space_frustrum]
    return [pos[0:3] / pos[3] for pos in global_space_frustrum]


def _load_model(obj_path):
    vertices = []
    uvs = []
    normals = []
    triangles = []
    with open(obj_path, 'r') as obj_file:
        for line in obj_file:
            split = line.split()
            if not split:
                continue
            data_type = split[0]
            data = [float(str_val) for str_val in split[1:]]
            if data_type == 'v':
                assert len(data) == 3
                vertices.append(data)
            elif data_type == 'vn':
                assert len(data) == 3
                normals.append(data)
            elif data_type == 'vt':
                assert len(data) == 2
                uvs.append(data)
            elif data_type == 'f':
                assert len(data) == 3
                data = [x - 1 for x in data]
                triangles.append(data)

    assert len(vertices) == len(uvs) == len(normals)
    vertices_vbo_data = np.concatenate(
        (np.array(vertices), np.array(normals), np.array(uvs)),
        axis=1)
    vertices_vbo = vbo.VBO(np.array(vertices_vbo_data, dtype=np.float32))
    indices_vbo = vbo.VBO(np.array(triangles, dtype=np.uint32),
                          target=GL.GL_ELEMENT_ARRAY_BUFFER)
    return vertices_vbo, indices_vbo


def _load_texture(cam_obj_texture_path):
    texture_data_bgr_flipped = cv2.imread(cam_obj_texture_path)
    texture_data_bgr = cv2.flip(texture_data_bgr_flipped, 0)
    texture_object = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_object)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                    texture_data_bgr.shape[1], texture_data_bgr.shape[0],
                    0, GL.GL_BGR, GL.GL_UNSIGNED_BYTE, texture_data_bgr)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return texture_object


_SCREEN_TEXTURE_W = 640
_SCREEN_TEXTURE_H = 480


def _allocate_empty_texture():
    texture_object = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_object)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                    _SCREEN_TEXTURE_W, _SCREEN_TEXTURE_H,
                    0, GL.GL_BGR, GL.GL_UNSIGNED_BYTE,
                    np.zeros((_SCREEN_TEXTURE_H, _SCREEN_TEXTURE_W, 3), dtype=np.uint8))
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return texture_object


def _configure_screen_frame_buffer(texture_object):
    frame_buffer_object = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, frame_buffer_object)
    GL.glFramebufferTexture(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, texture_object, 0)
    GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT0])
    assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    return frame_buffer_object


def _interpolate_transformation(t_1, t_2, coefficient):
    assert 0 <= coefficient < 1
    t_result = np.identity(4)
    t_result[0:3, 3] = t_1[0:3, 3] * (1 - coefficient) + t_2[0:3, 3] * coefficient
    rot_diff = t_2[0:3, 0:3].dot(t_1[0:3, 0:3].transpose())
    q_diff = quaternions.mat2quat(rot_diff)
    axis, angle = quaternions.quat2axangle(q_diff)
    rot_delta = quaternions.quat2mat(quaternions.axangle2quat(axis, angle * coefficient))
    rot_result = rot_delta.dot(t_1[0:3, 0:3])
    t_result[0:3, 0:3] = rot_result
    return t_result


def _interpolate_track(tracked_cam_track, tracked_cam_track_pos_float):
    assert 0 <= tracked_cam_track_pos_float <= len(tracked_cam_track) - 1
    if tracked_cam_track_pos_float == len(tracked_cam_track) - 1:
        return tracked_cam_track[-1]
    prev_idx = int(np.floor(tracked_cam_track_pos_float))
    next_idx = prev_idx + 1
    return _interpolate_transformation(tracked_cam_track[prev_idx], tracked_cam_track[next_idx],
                                       tracked_cam_track_pos_float - prev_idx)


def _build_view_matrix(camera_pos, camera_rot_mat):
    inv_view = np.identity(4)
    inv_view[0:3, 0:3] = camera_rot_mat
    inv_view[0:3, 3] = camera_pos
    view = np.linalg.inv(inv_view)
    return view


def _build_shaders():
    point_cloud_vertex_shader = shaders.compileShader(
        """
        #version 140
        uniform mat4 mvp;

        in vec3 position;
        in vec3 color;

        varying vec3 f_color;

        void main() {
            vec4 camera_space_position = mvp * vec4(position, 1.0);
            gl_Position = camera_space_position;
            f_color = color;
            gl_PointSize = 20 / gl_Position[3];
        }""",
        GL.GL_VERTEX_SHADER
    )
    color_interpolating_fragment_shader = shaders.compileShader(
        """
        #version 140
        varying vec3 f_color;

        out vec3 out_color;

        void main() {
            out_color = f_color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    line_vertex_shader = shaders.compileShader(
        """
        #version 140
        uniform mat4 mvp;
        in vec3 position;

        void main() {
            gl_Position = mvp * vec4(position, 1);
        }""",
        GL.GL_VERTEX_SHADER
    )
    line_fragment_shader = shaders.compileShader(
        """
        #version 140
        uniform vec3 color;

        void main() {
            gl_FragColor = vec4(color, 1);
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    camera_vertex_shader = shaders.compileShader(
        """
        #version 140
        uniform mat4 mvp;
        uniform mat3 normal_transformation;

        in vec3 position;
        in vec3 normal;
        in vec2 uv;

        varying vec3 f_normal;
        varying vec2 f_uv;

        void main() {
            gl_Position = mvp * vec4(position, 1);
            f_normal = normalize(normal_transformation * normal);
            f_uv = uv;
        }""",
        GL.GL_VERTEX_SHADER
    )
    camera_fragment_shader = shaders.compileShader(
        """
        #version 140
        uniform sampler2D tex;
        uniform sampler2D screen_tex;

        varying vec3 f_normal;
        varying vec2 f_uv;

        void main() {
            vec2 screen_uvs_lb = vec2(0.113281, 0.0313);
            vec2 screen_uvs_lt = vec2(0.113281, 0.1485);
            vec2 screen_uvs_rt = vec2(0.28125, 0.1485);
            vec2 screen_uvs_rb = vec2(0.28125, 0.0313);
            float screen_uvs_w = screen_uvs_rt[0] - screen_uvs_lt[0];
            float screen_uvs_h = screen_uvs_lt[1] - screen_uvs_lb[1];

            vec2 screen_uv = (f_uv - screen_uvs_lb);
            screen_uv[0] /= screen_uvs_w;
            screen_uv[1] /= screen_uvs_h;

            vec3 texture_color = texture(tex, f_uv).rgb;
            vec3 screen_texture_color = texture(screen_tex, screen_uv).rgb;
            vec3 color;
            if (screen_uv[0] >= -0. && screen_uv[0] <= 1. &&
                screen_uv[1] >= -0. && screen_uv[1] <= 1.) {
                color = screen_texture_color;
            } else {
                color = texture_color;
            }

            vec3 normal = normalize(f_normal);
            float cos_to_camera = dot(normal, vec3(0, 0, 1));
            float color_multiplier = max(cos_to_camera, 0);
            gl_FragColor = vec4(color_multiplier * color, 1);
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return namedtuple('CameraTrackRendererShaders', 'point_cloud line camera')(
        shaders.compileProgram(
            point_cloud_vertex_shader, color_interpolating_fragment_shader
        ),
        shaders.compileProgram(
            line_vertex_shader, line_fragment_shader
        ),
        shaders.compileProgram(
            camera_vertex_shader, camera_fragment_shader
        )
    )


def _to_transformation_mat(pose):
    res = np.eye(4)
    res[0:3, 0:3] = pose.r_mat
    res[0:3, 3] = pose.t_vec
    return res


class CameraTrackRenderer:

    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose],
                 point_cloud: data3d.PointCloud):
        """
        Initialize CameraTrackRenderer. Load camera model, create buffer objects, load textures,
        compile shaders, e.t.c.

        :param cam_model_files: path to camera model obj file and texture. The model consists of
        triangles with per-point uv and normal attributes
        :param tracked_cam_parameters: tracked camera field of view and aspect ratio. To be used
        for building tracked camera frustrum
        :param point_cloud: colored point cloud
        """

        tracked_cam_track = np.array([_to_transformation_mat(pose) for pose in tracked_cam_track])

        _transform_track(tracked_cam_track, point_cloud)

        camera_vertex_vbo, camera_index_vbo = _load_model(cam_model_files[0])
        self._texture_objects = namedtuple('TextureObjects', 'camera camera_screen')(
            _load_texture(cam_model_files[1]), _allocate_empty_texture()
        )
        self._camera_screen_frame_buffer = \
            _configure_screen_frame_buffer(self._texture_objects.camera_screen)

        self._tracked_cam_projection = _projection_matrix(
            tracked_cam_parameters.fov_y, tracked_cam_parameters.aspect_ratio, 0.1, 30)

        self._tracked_cam_track = tracked_cam_track

        BufferObjects = namedtuple(
            'BufferObjects', 'camera_vertices camera_indices frustrum point_cloud track'
        )
        self._buffer_objects = BufferObjects(
            camera_vertex_vbo, camera_index_vbo,
            vbo.VBO(
                np.array(_generate_frustrum(self._tracked_cam_projection), dtype=np.float32)
            ),
            vbo.VBO(np.array(np.stack(
                (point_cloud.points, point_cloud.colors)
            ), dtype=np.float32)),
            vbo.VBO(
                np.array([cam_pos[0:3, 3] for cam_pos in tracked_cam_track], dtype=np.float32)
            )
        )

        self._shaders = _build_shaders()

        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def display(self, camera_tr_vec, camera_rot_mat, camera_fov_y, tracked_cam_track_pos_float):
        """
        Draw everything with specified render camera position, projection parameters and 
        tracked camera position

        :param camera_tr_vec: vec3 position of render camera in global space
        :param camera_rot_mat: mat3 rotation matrix of render camera in global space
        :param camera_fov_y: render camera field of view. To be used for building a projection
        matrix. Use glutGet to calculate current aspect ratio
        :param tracked_cam_track_pos_float: a frame in which tracked camera
        model and frustrum should be drawn (see tracked_cam_track_pos for basic task)
        :return: returns nothing
        """

        # a frame in which a tracked camera model and frustrum should be drawn
        # without interpolation
        # tracked_cam_track_pos = int(tracked_cam_track_pos_float)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        view = _build_view_matrix(camera_tr_vec, camera_rot_mat)
        projection = _projection_matrix(
            camera_fov_y,
            GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH) / float(GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)),
            0.1,
            100.0)

        view_proj = projection.dot(view)

        camera_model_mat = _interpolate_track(self._tracked_cam_track, tracked_cam_track_pos_float)
        camera_normal_transformation = view.dot(camera_model_mat)[0:3, 0:3]
        camera_mvp = view_proj.dot(camera_model_mat)

        self._render_screen_texture(camera_model_mat)

        self._render_camera_model(camera_mvp, camera_normal_transformation)
        self._render_frustrum(camera_mvp)
        self._render_tracked_cam_track(view_proj)
        self._render_point_cloud(view_proj)

        GLUT.glutSwapBuffers()

    def _render_camera_model(self, mvp, camera_normal_transformation):
        shaders.glUseProgram(self._shaders.camera)
        try:
            self._buffer_objects.camera_vertices.bind()
            self._buffer_objects.camera_indices.bind()
            position_loc = None
            normal_loc = None
            uv_loc = None
            try:
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(self._shaders.camera, 'mvp'),
                    1, True, mvp)
                GL.glUniformMatrix3fv(
                    GL.glGetUniformLocation(self._shaders.camera, 'normal_transformation'),
                    1, True, camera_normal_transformation)
                GL.glUniform3fv(
                    GL.glGetUniformLocation(self._shaders.camera, 'color'),
                    1, np.array([1, 0.3, 0], dtype=np.float32))

                vertex_size = 3 * 4 + 3 * 4 + 2 * 4

                position_loc = GL.glGetAttribLocation(self._shaders.camera, 'position')
                GL.glEnableVertexAttribArray(position_loc)
                GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                         False, vertex_size,
                                         self._buffer_objects.camera_vertices)

                normal_loc = GL.glGetAttribLocation(self._shaders.camera, 'normal')
                GL.glEnableVertexAttribArray(normal_loc)
                GL.glVertexAttribPointer(normal_loc, 3, GL.GL_FLOAT,
                                         False, vertex_size,
                                         self._buffer_objects.camera_vertices + 3 * 4)

                uv_loc = GL.glGetAttribLocation(self._shaders.camera, 'uv')
                GL.glEnableVertexAttribArray(uv_loc)
                GL.glVertexAttribPointer(uv_loc, 2, GL.GL_FLOAT,
                                         False, vertex_size,
                                         self._buffer_objects.camera_vertices + 3 * 4 + 3 * 4)

                GL.glUniform1i(GL.glGetUniformLocation(self._shaders.camera, 'tex'), 0)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_objects.camera)

                GL.glUniform1i(GL.glGetUniformLocation(self._shaders.camera, 'screen_tex'), 1)
                GL.glActiveTexture(GL.GL_TEXTURE0 + 1)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_objects.camera_screen)

                GL.glDrawElements(GL.GL_TRIANGLES,
                                  3 * self._buffer_objects.camera_indices.data.shape[0],
                                  GL.GL_UNSIGNED_INT,
                                  self._buffer_objects.camera_indices)
            finally:
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                if uv_loc is not None:
                    GL.glDisableVertexAttribArray(uv_loc)
                if normal_loc is not None:
                    GL.glDisableVertexAttribArray(normal_loc)
                if position_loc is not None:
                    GL.glDisableVertexAttribArray(position_loc)
                self._buffer_objects.camera_indices.unbind()
                self._buffer_objects.camera_vertices.unbind()
        finally:
            shaders.glUseProgram(0)

    def _render_frustrum(self, mvp):
        shaders.glUseProgram(self._shaders.line)
        try:
            self._buffer_objects.frustrum.bind()
            position_loc = None
            try:
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(self._shaders.line, 'mvp'),
                    1, True, mvp)
                GL.glUniform3fv(
                    GL.glGetUniformLocation(self._shaders.line, 'color'),
                    1, np.array([1, 1, 0], dtype=np.float32))

                position_loc = GL.glGetAttribLocation(self._shaders.line, 'position')
                GL.glEnableVertexAttribArray(position_loc)
                GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                         False, 0,
                                         self._buffer_objects.frustrum)
                GL.glDrawArrays(GL.GL_LINES, 0, self._buffer_objects.frustrum.data.shape[0])
            finally:
                if position_loc is not None:
                    GL.glDisableVertexAttribArray(position_loc)
                self._buffer_objects.frustrum.unbind()
        finally:
            shaders.glUseProgram(0)

    def _render_tracked_cam_track(self, view_proj):
        shaders.glUseProgram(self._shaders.line)
        try:
            self._buffer_objects.track.bind()
            position_loc = None
            try:
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(self._shaders.line, 'mvp'),
                    1, True, view_proj)
                GL.glUniform3fv(
                    GL.glGetUniformLocation(self._shaders.line, 'color'),
                    1, np.array([1, 1, 1], dtype=np.float32))

                position_loc = GL.glGetAttribLocation(self._shaders.line, 'position')
                GL.glEnableVertexAttribArray(position_loc)
                GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                         False, 0,
                                         self._buffer_objects.track)
                GL.glDrawArrays(GL.GL_LINE_STRIP, 0, self._buffer_objects.track.data.shape[0])
            finally:
                if position_loc is not None:
                    GL.glDisableVertexAttribArray(position_loc)
                self._buffer_objects.track.unbind()
        finally:
            shaders.glUseProgram(0)

    def _render_point_cloud(self, mvp):
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        shaders.glUseProgram(self._shaders.point_cloud)
        try:
            self._buffer_objects.point_cloud.bind()
            position_loc = None
            color_loc = None
            try:
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(self._shaders.point_cloud, 'mvp'),
                    1, True, mvp)
                point_cloud_size = self._buffer_objects.point_cloud.data.shape[1]

                position_loc = GL.glGetAttribLocation(self._shaders.point_cloud, 'position')
                GL.glEnableVertexAttribArray(position_loc)
                GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                         False, 0,
                                         self._buffer_objects.point_cloud)

                color_loc = GL.glGetAttribLocation(self._shaders.point_cloud, 'color')
                GL.glEnableVertexAttribArray(color_loc)
                GL.glVertexAttribPointer(
                    color_loc, 3, GL.GL_FLOAT,
                    False, 0,
                    self._buffer_objects.point_cloud + 3 * 4 * point_cloud_size
                )
                GL.glDrawArrays(GL.GL_POINTS, 0, point_cloud_size)
            finally:
                if color_loc is not None:
                    GL.glDisableVertexAttribArray(color_loc)
                if position_loc is not None:
                    GL.glDisableVertexAttribArray(position_loc)
                self._buffer_objects.point_cloud.unbind()
        finally:
            shaders.glUseProgram(0)
        GL.glDisable(GL.GL_PROGRAM_POINT_SIZE)

    def _render_screen_texture(self, inv_view_mat):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._camera_screen_frame_buffer)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glViewport(0, 0, _SCREEN_TEXTURE_W, _SCREEN_TEXTURE_H)
        mvp = self._tracked_cam_projection.dot(np.linalg.inv(inv_view_mat))
        self._render_point_cloud(mvp)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glViewport(0, 0, GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH),
                      GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT))
