# Camera_Pose_Estimation_and_AR
Camera Pose Estimation and AR program
## Demo video:
[[![output_wonoo_video_ok](https://img.youtube.com/vi/VIDEO_ID/0.jpg)]](https://github.com/yunee19/Camera_Pose_Estimation_and_AR/assets/133479803/79fb913a-6667-42a4-93fa-d63f1643f23e)

이 프로그램은 카메라 캘리브레이션과 카메라 자세 추정을 수행하여 카메라 영상에 AR(증강 현실) 물체를 표시합니다. 주요 단계와 결과는 다음과 같습니다:

1. **카메라 캘리브레이션**: 프로그램은 체스보드 패턴을 사용하여 카메라를 캘리브레이션합니다. 캘리브레이션은 카메라의 내부 파라미터(카메라 행렬)와 왜곡 계수를 계산하는 것을 포함합니다. 이를 위해 OpenCV의 `calibrateCamera` 함수를 사용합니다.

2. **카메라 자세 추정**: 캘리브레이션이 완료되면 프로그램은 카메라 자세(회전 및 이동 벡터)를 추정합니다. 이를 위해 OpenCV의 `solvePnPRansac` 함수를 사용하여 체스보드의 코너 점을 3D 객체로 변환하고 이에 대한 회전 및 이동 벡터를 찾습니다.

3. **AR 물체 표시**: 추정된 카메라 자세를 사용하여 3D 축을 카메라 영상에 투영하고, 이를 통해 AR 물체를 시각화합니다. 이를 위해 OpenCV의 `projectPoints` 함수를 사용하여 3D 객체의 점을 이미지 평면으로 투영합니다.

이 프로그램은 카메라가 올바르게 인식되지 않거나 코너를 찾지 못할 때 오류가 발생할 수 있습니다. 이를 해결하기 위해서는 카메라 연결 상태를 확인하고, 체스보드 패턴이 올바르게 감지되었는지 확인해야 합니다. 또한 캘리브레이션 및 자세 추정 과정에서 발생하는 오류를 적절히 처리해야 합니다.
