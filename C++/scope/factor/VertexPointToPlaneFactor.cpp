#include <glog/logging.h>

#include <scope/factor/VertexPointToPlaneFactor.h>

namespace scope {
VertexPointToPlaneFactor::VertexPointToPlaneFactor(
    int pose, int vParam, const Matrix3X &vDirs, const Vector3 &v,
    const Scalar &sigma, const Scalar &eps, const Vector3 &measurement,
    const Vector3 &normal, const Scalar &confidence,
    const std::string &name, int index, bool active)
    : DepthCameraFactor({pose}, {}, {}, {vParam}, sigma, eps, measurement,
                        confidence, name, index, active),
      mvDirs(vDirs),
      mv(v),
      mNormal(normal.normalized()) {}

int VertexPointToPlaneFactor::evaluate(const AlignedVector<Pose> &poses,
                                       const AlignedVector<VectorX> &shapes,
                                       const AlignedVector<Matrix3> &joints,
                                       const AlignedVector<VectorX> &params,
                                       Factor::Evaluation &base_eval) const {
  auto &eval = dynamic_cast<Evaluation &>(base_eval);
  eval.clear();

  const auto &pose = mvPoses[0];
  assert(pose >= 0 && pose < poses.size());

  if (pose < 0 || pose >= poses.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &vertexParam = mvParams[0];
  assert(vertexParam >= 0 && vertexParam < params.size());

  if (vertexParam < 0 || vertexParam > params.size()) {
    LOG(ERROR) << "The parameter must be valid." << std::endl;

    exit(-1);
  }

  evaluate(poses[pose], params[vertexParam], mMeasurement, mNormal, eval);

  eval.status = Status::VALID;

  return 0;
}

int VertexPointToPlaneFactor::linearize(const AlignedVector<Pose> &poses,
                                        const AlignedVector<VectorX> &shapes,
                                        const AlignedVector<Matrix3> &joints,
                                        const AlignedVector<VectorX> &params,
                                        const Factor::Evaluation &base_eval,
                                        Factor::Linearization &base_lin) const {
  auto &eval = dynamic_cast<const Evaluation &>(base_eval);
  auto &lin = dynamic_cast<Linearization &>(base_lin);

  lin.clear();

  assert(eval.status == Status::VALID);

  if (eval.status != Status::VALID) {
    LOG(ERROR) << "The evaluation must be valid." << std::endl;

    exit(-1);
  }

  const auto &pose = mvPoses[0];
  assert(pose >= 0 && pose < poses.size());

  if (pose < 0 || pose >= poses.size()) {
    LOG(ERROR) << "The pose must be valid." << std::endl;

    exit(-1);
  }

  const auto &vertexParam = mvParams[0];
  assert(vertexParam >= 0 && vertexParam < params.size());

  if (vertexParam < 0 || vertexParam > params.size()) {
    LOG(ERROR) << "The parameter must be valid." << std::endl;

    exit(-1);
  }

  lin.jacobians[0].resize(1);
  lin.jacobians[3].resize(1);

  linearize(poses[pose], params[vertexParam], mMeasurement, mNormal, eval, lin);

  lin.status = Status::VALID;

  return 0;
}

int VertexPointToPlaneFactor::evaluate(const Pose &pose,
                                       const VectorX &vertexParam,
                                       const Vector3 &measurement,
                                       const Vector3 &normal,
                                       Evaluation &eval) const {
  eval.vertex = mv;
  eval.vertex.noalias() += mvDirs * vertexParam;

  eval.point3D = pose.t;
  eval.point3D.noalias() += pose.R * eval.vertex;
  eval.normal = normal;

  eval.error.resize(1);
  eval.error[0] = eval.normal.dot(eval.point3D - measurement);
  eval.squaredErrorNorm = eval.error[0] * eval.error[0];

  eval.f = mCon * eval.squaredErrorNorm;

  return 0;
}

int VertexPointToPlaneFactor::linearize(const Pose &pose,
                                        const VectorX &vertexParam,
                                        const Vector3 &measurement,
                                        const Vector3 &normal,
                                        const Evaluation &eval,
                                        Linearization &lin) const {
  assert(lin.jacobians[0].size() == 1);
  assert(lin.jacobians[3].size() >= 1);

  Matrix<1, 6> JPose;
  Matrix<3, 6> JPoseVec;

  JPoseVec.setZero();

  JPoseVec(0, 1) = eval.point3D[2];
  JPoseVec(0, 2) = -eval.point3D[1];
  JPoseVec(1, 0) = -eval.point3D[2];
  JPoseVec(1, 2) = eval.point3D[0];
  JPoseVec(2, 0) = eval.point3D[1];
  JPoseVec(2, 1) = -eval.point3D[0];

  JPoseVec.block<3, 3>(0, 3).setIdentity();

  JPose.noalias() = eval.normal.transpose() * JPoseVec;

  auto &JPoseStorage = lin.jacobians[0][0];
  JPoseStorage = JPose;

  Matrix<1, Eigen::Dynamic> JVertex;
  JVertex.resize(1, mvDirs.cols());
  JVertex.noalias() = eval.normal.transpose() * pose.R * mvDirs;

  auto &JVertexStorage = lin.jacobians[3][0];
  JVertexStorage = JVertex;

  auto &gPose = lin.gPose;
  auto &gVertex = lin.gVertex;

  gPose.noalias() = JPose.transpose() * eval.error[0];
  gVertex.noalias() = JVertex.transpose() * eval.error[0];

  JPoseStorage *= mSqrtCon;
  JVertexStorage *= mSqrtCon;

  gPose *= mCon;
  gVertex *= mCon;

  return 0;
}
}  // namespace scope
