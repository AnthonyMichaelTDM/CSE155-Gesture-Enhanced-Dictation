from enum import StrEnum
from typing import Optional
import cv2
from cv2.typing import MatLike, Point, Rect

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_thumb_landmarks(
    image: MatLike, thumb_mcp: Point, thumb_ip: Point, thumb_tip: Point
) -> MatLike:
    cv2.line(image, tuple(thumb_mcp), tuple(thumb_ip), BLACK, 6)
    cv2.line(
        image,
        tuple(thumb_mcp),
        tuple(thumb_ip),
        WHITE,
        2,
    )
    cv2.line(image, tuple(thumb_ip), tuple(thumb_tip), BLACK, 6)
    cv2.line(
        image,
        tuple(thumb_ip),
        tuple(thumb_tip),
        WHITE,
        2,
    )
    return image


def draw_finger_landmarks(
    image: MatLike,
    finger_mcp: Point,
    finger_pip: Point,
    finger_dip: Point,
    finger_tip: Point,
) -> MatLike:
    cv2.line(image, finger_mcp, finger_pip, BLACK, 6)
    cv2.line(
        image,
        finger_mcp,
        finger_pip,
        WHITE,
        2,
    )
    cv2.line(image, finger_pip, finger_dip, BLACK, 6)
    cv2.line(
        image,
        finger_pip,
        finger_dip,
        WHITE,
        2,
    )
    cv2.line(image, finger_dip, finger_tip, BLACK, 6)
    cv2.line(
        image,
        finger_dip,
        finger_tip,
        WHITE,
        2,
    )
    return image


def draw_palm_landmarks(
    image: MatLike,
    wrist: Point,
    thumb_cmc: Point,
    thumb_mcp: Point,
    index_finger_mcp: Point,
    middle_finger_mcp: Point,
    ring_finger_mcp: Point,
    little_finger_mcp: Point,
) -> MatLike:
    cv2.line(image, wrist, thumb_cmc, BLACK, 6)
    cv2.line(
        image,
        wrist,
        thumb_cmc,
        WHITE,
        2,
    )
    cv2.line(image, thumb_cmc, thumb_mcp, BLACK, 6)
    cv2.line(
        image,
        thumb_cmc,
        thumb_mcp,
        WHITE,
        2,
    )
    cv2.line(image, thumb_mcp, index_finger_mcp, BLACK, 6)
    cv2.line(
        image,
        thumb_mcp,
        index_finger_mcp,
        WHITE,
        2,
    )
    cv2.line(image, index_finger_mcp, middle_finger_mcp, BLACK, 6)
    cv2.line(
        image,
        index_finger_mcp,
        middle_finger_mcp,
        WHITE,
        2,
    )
    cv2.line(image, middle_finger_mcp, ring_finger_mcp, BLACK, 6)
    cv2.line(
        image,
        middle_finger_mcp,
        ring_finger_mcp,
        WHITE,
        2,
    )
    cv2.line(image, ring_finger_mcp, little_finger_mcp, BLACK, 6)
    cv2.line(
        image,
        ring_finger_mcp,
        little_finger_mcp,
        WHITE,
        2,
    )
    cv2.line(image, little_finger_mcp, wrist, BLACK, 6)
    cv2.line(
        image,
        little_finger_mcp,
        wrist,
        WHITE,
        2,
    )

    return image


def draw_landmarks(
    image: MatLike,
    landmark_point: list[Point],
) -> MatLike:

    if len(landmark_point) > 0:
        # Thumb
        draw_thumb_landmarks(
            image, landmark_point[2], landmark_point[3], landmark_point[4]
        )

        # Index finger
        draw_finger_landmarks(
            image,
            landmark_point[5],
            landmark_point[6],
            landmark_point[7],
            landmark_point[8],
        )

        # Middle finger
        draw_finger_landmarks(
            image,
            landmark_point[9],
            landmark_point[10],
            landmark_point[11],
            landmark_point[12],
        )

        # Ring finger
        draw_finger_landmarks(
            image,
            landmark_point[13],
            landmark_point[14],
            landmark_point[15],
            landmark_point[16],
        )

        # Little finger
        draw_finger_landmarks(
            image,
            landmark_point[17],
            landmark_point[18],
            landmark_point[19],
            landmark_point[20],
        )

        # Palm
        draw_palm_landmarks(
            image,
            landmark_point[0],
            landmark_point[1],
            landmark_point[2],
            landmark_point[5],
            landmark_point[9],
            landmark_point[13],
            landmark_point[17],
        )

    # draw dots on the landmarks / key points
    # making the finger tips bigger
    FINGER_TIPS = [4, 8, 12, 16, 20]
    for index, landmark in enumerate(landmark_point):
        cv2.circle(
            image,
            (landmark[0], landmark[1]),
            8 if ((not index == 0) and index % 4 == 0) else 5,
            WHITE,
            -1,
        )
        cv2.circle(
            image,
            (landmark[0], landmark[1]),
            8 if ((not index == 0) and index % 4 == 0) else 5,
            BLACK,
            1,
        )

    return image


class BoundingBoxType(StrEnum):
    """
    The type of bounding box to draw
    """

    Dwell = "dwell"
    Success = "success"
    Default = "default"


def draw_bounding_rect(
    use_brect: bool,
    image: MatLike,
    brect: cv2.typing.Rect,
    val: BoundingBoxType = BoundingBoxType.Default,
) -> MatLike:
    if use_brect:
        x, y, w, h = brect
        pt1 = (x, y)
        pt2 = (x + w, y + h)

        if val == BoundingBoxType.Dwell:
            # Outer rectangle (orange)
            cv2.rectangle(image, pt1, pt2, (0, 165, 255), 1)
        elif val == BoundingBoxType.Success:
            # Outer rectangle (green)
            cv2.rectangle(image, pt1, pt2, (0, 255, 0), 1)
        else:
            # Outer rectangle (black)
            cv2.rectangle(image, pt1, pt2, BLACK, 1)

    return image


def draw_info_text(
    image: MatLike, brect: cv2.typing.Rect, handedness: str, hand_sign_text: str
) -> MatLike:
    x, y, w, _ = brect
    pt1 = (x, y)

    cv2.rectangle(image, pt1, (x + w, y - 22), BLACK, -1)

    info_text = handedness
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text
    cv2.putText(
        image,
        info_text,
        (x + 5, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        WHITE,
        1,
        cv2.LINE_AA,
    )

    return image


def draw_info(
    image: MatLike, fps: float, keypoint_training_mode: bool, number: Optional[int]
) -> MatLike:
    cv2.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        BLACK,
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    if not keypoint_training_mode:
        return image

    cv2.putText(
        image,
        "MODE: Keypoint Training",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        WHITE,
        1,
        cv2.LINE_AA,
    )

    if number is not None:
        cv2.putText(
            image,
            f"Currently training for class {number}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            1,
            cv2.LINE_AA,
        )

    return image
