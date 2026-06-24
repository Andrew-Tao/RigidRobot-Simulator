import numpy as np

class CableDrivenForce:
    def __init__(self, control_input: callable, hole_offset: np.ndarray):
        self.control_input = control_input
        self.hole_offset = np.asarray(hole_offset, dtype=float)
        # Accepted shapes:
        #   (C, 3)    — uniform offset applied to every disk
        #   (N, C, 3) — per-disk offset; disk i uses hole_offset[i]
        self.cable_force_colletion = [] # TODO: for debug

    def _offset(self, disk_index: int, cable_index: int) -> np.ndarray:
        if self.hole_offset.ndim == 2:
            return self.hole_offset[cable_index]
        return self.hole_offset[disk_index, cable_index]

    def compute_force_collection(self, slender_robot, time):

        n = len(slender_robot.robots)
        n_cables = self.hole_offset.shape[0] if self.hole_offset.ndim == 2 else self.hole_offset.shape[1]
        force_collection = np.zeros((n, 6))

        for i in range(n):
            force_on_single_robot = np.zeros(6)
            orientation_Q = slender_robot.robots[i].posture[:3, :3]

            for cable_index in range(n_cables):

                offset_i  = self._offset(i, cable_index)  # body-frame hole offset
              
                position  = slender_robot.robots[i].posture[:3, 3] + orientation_Q @ offset_i

                if i != 0:
                    Q_prev = slender_robot.robots[i-1].posture[:3, :3]
                    anchor_point_before = slender_robot.robots[i-1].posture[:3, 3] + Q_prev @ self._offset(i-1, cable_index)
                else:
                    anchor_point_before = self._offset(0, cable_index)   # base hole at origin, fixed

                if i != n - 1:
                    Q_next = slender_robot.robots[i+1].posture[:3, :3]
                    anchor_point_after = slender_robot.robots[i+1].posture[:3, 3] + Q_next @ self._offset(i+1, cable_index)
                else:
                    anchor_point_after = None

                #-------------------- Compute the Forces in local frame -------------------
                anchor_point_local_before = orientation_Q.T @ (anchor_point_before - position)
                if np.linalg.norm(anchor_point_local_before) > 1e-6:
                    unit_anchor_vector_local_before = anchor_point_local_before / np.linalg.norm(anchor_point_local_before)
                else:
                    unit_anchor_vector_local_before = np.zeros(3)

                if anchor_point_after is not None:
                    anchor_point_local_after = orientation_Q.T @ (anchor_point_after - position)
                    if np.linalg.norm(anchor_point_local_after) > 1e-6:
                        unit_anchor_vector_local_after = anchor_point_local_after / np.linalg.norm(anchor_point_local_after)
                    else:
                        unit_anchor_vector_local_after = np.zeros(3)
                else:
                    unit_anchor_vector_local_after = np.zeros(3)

                tension = self.control_input(time)[cable_index]
                linear_force_before = tension * unit_anchor_vector_local_before
                linear_force_after  = tension * unit_anchor_vector_local_after
                linear_total_on_single_robot = linear_force_before + linear_force_after

                # --------------------- Compute Torque in Local Frame ----------------------
                force_arm_before = offset_i + np.array([0.0, 0.0, -slender_robot.robots[i].thickness * 0.5])
                force_arm_after  = offset_i + np.array([0.0, 0.0,  slender_robot.robots[i].thickness * 0.5])
                torque_before = np.cross(force_arm_before, linear_force_before)
                torque_after  = np.cross(force_arm_after,  linear_force_after)
                torque_total_on_single_robot = torque_before + torque_after

                # --------------------- Total Force by one Cable ----------------------------
                force_on_single_robot += np.hstack((linear_total_on_single_robot, torque_total_on_single_robot))

            # Bug 3 fix: collect tip disk (i == n-1) instead of hardcoded i == 2
            if i == n - 1:
                self.cable_force_colletion.append(force_on_single_robot)

            force_collection[i] = force_on_single_robot

        return force_collection


class GravityForce:
    def __init__(self):

       self.gravity_constant = 9.81

    def compute_force_collection(self, slender_robot, time):

        n = len(slender_robot.robots)
        force_collection = np.zeros((n, 6))  # Bug 1 fixed: np.zeros needs a tuple shape

        for i in range(n):
            force_on_single_robot = np.zeros(6)

            Q = slender_robot.robots[i].posture[:3, :3]
            
            gravity_force_in_global_frame = np.array([0.0, 0.0, - slender_robot.robots[i].mass_matrix[0,0] * self.gravity_constant])
            gravity_force_in_local_frame = Q.T @ gravity_force_in_global_frame
            force_on_single_robot = np.hstack((gravity_force_in_local_frame, np.zeros(3)))

            force_collection[i] = force_on_single_robot
        return force_collection
