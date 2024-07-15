from warnings import warn
from typing import Union

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch as tp


class OptImage:
    def __init__(self, wave_length: float, size: float, shape: tuple) -> None:
        self.wave_length = wave_length
        self.size = size
        self.shape = shape
        self.shifted = False

    def __str__(self) -> str:
        return f"Image(\n\tlambda: ({self.wave_length})m\n\tpixel width: ({self.get_pixel_width():.3e})m\n\tsize: ({self.size}, {self.size})m\n\tshape: ({self.shape[0]}, {self.shape[1]})pixels\n)"

    def get_size(self) -> float:
        """
        Returns the current image size in meters [m]
        """
        return self.size

    def get_shape(self) -> tuple:
        """
        Returns the current image size in pixels [pixels]
        """
        return self.shape

    def get_wave_length(self) -> float:
        """
        Returns the specified wavelength in meters [m]
        """
        return self.wave_length

    def get_pixel_width(self) -> float:
        """
        Returns the current width of the pixel in meters [m]
        """
        return self.size / self.shape[0]

    def set_pixel_width(self, width: float) -> None:
        """
        Sets the pixel width of the current image in meters [m]\n
        Parameters
        ----------
        `width` : float - pixel width in meters [m]
        """
        self.size = self.shape[0] * width

    def __add__(self): ...
    def __mul__(self): ...

    def append_amplitude(self): ...
    def append_phase(self): ...
    def set_amplitude(self): ...
    def set_phase(self): ...

    def get_amplitude(self): ...
    def get_phase(self): ...
    def get_real_part(self): ...
    def get_intensity(self): ...

    def fresnell(self): ...
    def pad_zeros(self): ...
    def imshow(self): ...
    def imwrite(self): ...


class NumpyOptImage(OptImage):
    def __init__(self, image: np.ndarray, wave_length: float, size: float) -> None:
        super().__init__(wave_length, size, image.shape)
        self.__image = image

    def __add__(self, other): ...
    def __mul__(self): ...

    def append_amplitude(self, src: np.ndarray) -> None:
        """
        Changes the amplitude of the current image\n
        Parameters
        ----------
        `src` : ndarray - amplitude change matrix
        """
        amplitude = self.get_amplitude()
        phase = self.get_phase()
        amplitude *= src
        self.__image = amplitude * np.exp(1j * phase)

    def append_phase(self, src: np.ndarray) -> None:
        """
        Changes the phase of the current image\n
        Parameters
        ----------
        `src` : ndarray - phase change matrix
        """
        amplitude = self.get_amplitude()
        phase = self.get_phase()
        phase += src
        self.__image = amplitude * np.exp(1j * phase)

    def get_amplitude(self) -> np.ndarray:
        """
        Returns the current value of the image amplitude
        """
        return np.abs(self.__image)

    def get_phase(self) -> np.ndarray:
        """
        Returns the current value of the image phase
        """
        return np.angle(self.__image)

    def get_real_part(self) -> np.ndarray:
        """
        Returns the current value of the real part of the image
        """
        return self.__image.real

    def get_intensity(self) -> np.ndarray:
        """
        Returns the current value of the image intensity
        """
        return self.get_real_part() ** 2

    def set_amplitude(self, amplitude: np.ndarray) -> None:
        """
        Changes the amplitude of the current image
        Parameters
        ----------
        `phase` : ndarray - new image amplitude matrix
        """
        phase = self.get_phase()
        self.__image = amplitude * np.exp(1j * phase)

    def set_phase(self, phase: np.ndarray) -> None:
        """
        Changes the phase of the current image
        Parameters
        ----------
        `phase` : ndarray - new image phase matrix
        """
        amplitude = self.get_amplitude()
        self.__image = amplitude * np.exp(1j * phase)

    def fresnell(self, distance: float) -> None:
        """
        Performs a discrete Fresnel transform for the current image\n
        Parameters
        ----------
        `distance` : float - distance to the detector
        """
        rows, cols = self.shape

        x = np.linspace(-self.size / 2, self.size / 2, rows)
        y = np.linspace(-self.size / 2, self.size / 2, cols)
        X, Y = np.meshgrid(x, y)

        exp = -np.exp(1j * np.pi * (X**2 + Y**2) / (self.wave_length * distance))
        exp_fft = np.exp(1j * np.pi * (X**2 + Y**2) / (self.wave_length * distance))

        fft = np.fft.fftshift(np.fft.fft2(self.__image * exp_fft))

        if self.shifted:
            fft = np.fft.fftshift(fft)

        self.shifted = not self.shifted
        self.__image = exp * fft

    def pad_zeros(self, shape: tuple) -> None:
        """
        Augments the current image with zeros up to the specified size\n
        Parameters
        ----------
        `shape` : float - size of the augmented image in pixels [pixels]
        """
        current_rows, current_cols = self.shape
        new_rows, new_cols = shape
        self.shape = shape

        top = (new_rows - current_rows) // 2
        bottom = top
        left = (new_cols - current_cols) // 2
        right = left

        self.__image = cv2.copyMakeBorder(self.__image, top, bottom, left, right, 0)

        if self.__image.shape != self.shape:
            self.shape = self.__image.shape
            warn(
                f"it is impossible to complete the image to the specified size, the current number of pixels: {self.shape}",
                category=RuntimeWarning,
                stacklevel=2,
            )

    def imshow(
        self, parts: list = ["intensity", "amplitude", "phase"], cmap: str = "jet"
    ) -> None:
        """
        Displays the current value of the intensity, amplitude, and phase of the image
        Parameters
        ----------
        `parts` : list - list of parameters to be displayed. Possible parameters: 'intensity', 'amplitude', 'phase'\n
        `cmap` : str - type of colormap used when displaying parts of the image. To see a list of available colormaps, use:
        >>> import matploltib.pyplot as plt
        >>> print(plt.colormaps)
        """
        num = len(parts)
        for i, part in enumerate(parts):
            plot_number = int(f"1{num}{i + 1}")

            plt.subplot(plot_number)
            plt.axis("off")
            plt.title(part)

            if part == "intensity":
                plt.imshow(self.get_intensity(), cmap=cmap)
            elif part == "amplitude":
                plt.imshow(self.get_amplitude(), cmap=cmap)
            elif part == "phase":
                plt.imshow(self.get_phase(), cmap=cmap)
            else:
                raise SyntaxError(
                    "parts must be: 'intensity' or 'amplitude' or 'phase'"
                )
        plt.show()

    def imwrite(self, path: str, cmap: str = "jet") -> None:
        """
        Writes the intensity, amplitude, and phase of the current image along the specified path with the specified extension\n
        Parameters
        ----------
        `path` : str - directory to which the current image is written (specified with image name and extension)\n
        `cmap`: str - type of colormap used when displaying parts of the image. To see a list of available colormaps, use:
        >>> import matploltib.pyplot as plt
        >>> print(plt.colormaps)
        """
        parts: list = ["intensity", "amplitude", "phase"]
        num = len(parts)
        for i, part in enumerate(parts):
            plot_number = int(f"1{num}{i + 1}")

            plt.subplot(plot_number)
            plt.axis("off")
            plt.title(part)

            if part == "intensity":
                plt.imshow(self.get_intensity(), cmap=cmap)
            elif part == "amplitude":
                plt.imshow(self.get_amplitude(), cmap=cmap)
            elif part == "phase":
                plt.imshow(self.get_phase(), cmap=cmap)
        plt.savefig(path)


class TorchOptImage(OptImage):
    def __init__(self, image: tp.Tensor, wave_length: float, size: float) -> None:
        if len(image.shape) > 3:
            raise ValueError(
                f"the image shape should have the form (x, y, z) or (x, y), but: {image.shape}"
            )
        super().__init__(wave_length, size, image.shape)
        self.__image = image

    def __add__(self, other):
        if isinstance(other, tp.nn.parameter.Parameter) or isinstance(other, tp.Tensor):
            if (
                other.shape[-1] != self.get_shape()[-1]
                or other.shape[-2] != self.get_shape()[-2]
            ):
                raise ValueError(
                    f"it is impossible to make an addition: the size of the parts to be folded must be the same, but: ({other.shape[-2]}, {other.shape[-1]}) and ({self.get_shape()[-2]}, {self.get_shape()[-1]})"
                )
            current_phase = self.get_phase()
            current_amplitude = self.get_amplitude()
            output = current_amplitude * tp.exp(1j * (current_phase + other))
            return TorchOptImage(output, self.get_wave_length(), self.get_size())

        elif isinstance(other, TorchLens):
            if (
                other.shape[-1] != self.get_shape()[-1]
                or other.shape[-2] != self.get_shape()[-2]
            ):
                raise ValueError(
                    f"it is impossible to make an addition: the size of the parts to be folded must be the same, but: ({other.shape[-2]}, {other.shape[-1]}) and ({self.get_shape()[-2]}, {self.get_shape()[-1]})"
                )
            other_phase = other.get_phase()
            current_phase = self.get_phase()
            current_amplitude = self.get_amplitude()
            output = current_amplitude * tp.exp(1j * (current_phase + other_phase))
            return TorchOptImage(output, self.get_wave_length(), self.get_size())

        else:
            raise SyntaxError(
                f"unsupported operand between types: {type(self)} and {type(other)}"
            )

    def __mul__(self, other):
        if isinstance(other, tp.nn.parameter.Parameter) or isinstance(other, tp.Tensor):
            if (
                other.shape[-1] != self.get_shape()[-1]
                or other.shape[-2] != self.get_shape()[-2]
            ):
                raise ValueError(
                    f"it is impossible to make a multyply: the size of the parts to be folded must be the same, but: ({other.shape[-2]}, {other.shape[-1]}) and ({self.get_shape()[-2]}, {self.get_shape()[-1]})"
                )
            current_phase = self.get_phase()
            current_amplitude = self.get_amplitude()
            output = (current_amplitude * other) * tp.exp(1j * (current_phase))
            return TorchOptImage(output, self.get_wave_length(), self.get_size())

        elif isinstance(other, TorchLens):
            if (
                other.shape[-1] != self.get_shape()[-1]
                or other.shape[-2] != self.get_shape()[-2]
            ):
                raise ValueError(
                    f"it is impossible to make a multyply: the size of the parts to be folded must be the same, but: ({other.shape[-2]}, {other.shape[-1]}) and ({self.get_shape()[-2]}, {self.get_shape()[-1]})"
                )
            current_phase = self.get_phase()
            other_amplitude = other.get_amplitude()
            current_amplitude = self.get_amplitude()
            output = (current_amplitude * other_amplitude) * tp.exp(
                1j * (current_phase)
            )
            return TorchOptImage(output, self.get_wave_length(), self.get_size())

        else:
            raise SyntaxError(
                f"unsupported operand between types: {type(self)} and {type(other)}"
            )

    def append_amplitude(self, src: tp.Tensor) -> None:
        """
        Changes the amplitude of the current image\n
        Parameters
        ----------
        `src` : Tensor - amplitude change matrix
        """
        current_phase = self.get_phase()
        current_amplitude = self.get_amplitude()
        self.__image = (current_amplitude + src) * tp.exp(1j * current_phase)

    def append_phase(self, src: tp.Tensor) -> None:
        """
        Changes the phase of the current image\n
        Parameters
        ----------
        `src` : Tensor - phase change matrix
        """
        current_phase = self.get_phase()
        current_amplitude = self.get_amplitude()
        self.__image = current_amplitude * tp.exp(1j * (current_phase + src))

    def get_amplitude(self) -> tp.Tensor:
        """
        Returns the current value of the image amplitude
        """
        return self.__image.abs()

    def get_phase(self) -> tp.Tensor:
        """
        Returns the current value of the image phase
        """
        return self.__image.abs()

    def get_real_part(self) -> tp.Tensor:
        """
        Returns the current value of the real part of the image
        """
        return self.__image.real

    def get_intensity(self) -> tp.Tensor:
        """
        Returns the current value of the image intensity
        """
        return self.get_real_part() ** 2

    def set_amplitude(self, amplitude: tp.Tensor) -> None:
        """
        Changes the amplitude of the current image
        Parameters
        ----------
        `phase` : Tensor - new image amplitude matrix
        """
        current_phase = self.get_phase()
        self.__image = amplitude * tp.exp(1j * current_phase)

    def set_phase(self, phase: tp.Tensor) -> None:
        """
        Changes the phase of the current image
        Parameters
        ----------
        `phase` : Tensor - new image phase matrix
        """
        current_amplitude = self.get_amplitude()
        self.__image = phase * tp.exp(1j * current_amplitude)

    def fresnell(self, distance: float) -> None:
        """
        Performs a discrete Fresnel transform for the current image\n
        Parameters
        ----------
        `distance` : float - distance to the detector
        """
        new_image = tp.empty(
            self.get_shape(), dtype=tp.complex64, device=self.__image.device
        )
        for i, channel in enumerate(self.__image):
            rows, cols = channel.shape

            x = tp.linspace(-self.size / 2, self.size / 2, rows)
            y = tp.linspace(-self.size / 2, self.size / 2, cols)
            X, Y = tp.meshgrid(x, y, indexing="ij")
            X = X.to(self.__image.device)
            Y = Y.to(self.__image.device)

            exp = -tp.exp(1j * tp.pi * (X**2 + Y**2) / (self.wave_length * distance))
            exp_fft = tp.exp(1j * tp.pi * (X**2 + Y**2) / (self.wave_length * distance))

            fft = tp.fft.fftshift(tp.fft.fft2(channel * exp_fft))

            if self.shifted:
                fft = tp.fft.fftshift(fft)

            new_image[i] = exp * fft

        self.shifted = not self.shifted
        self.__image = new_image

    def forward_fourier(self) -> None:
        self.__image = tp.fft.fftshift(tp.fft.fft2(self.__image))

    def lens_like_fourier(self) -> None:
        self.__image = tp.fft.fft2(tp.fft.fft2(self.__image))

    def pad_zeros(self, shape: tuple) -> None:
        """
        Augments the current image with zeros up to the specified size\n
        Parameters
        ----------
        `shape` : float - size of the augmented image in pixels [pixels]
        """
        current_rows, current_cols = self.__image[0].shape
        new_cols, new_rows = shape
        top = (new_rows - current_rows) // 2
        bottom = top
        left = (new_cols - current_cols) // 2
        right = left

        if (new_rows - current_rows) % 2 != 0 or (new_cols - current_cols) % 2 != 0:
            shape = (shape[0] - 1, shape[1] - 1)
            self.shape = (self.__image.shape[0], shape[0], shape[1])
            warn(
                f"it is impossible to complete the image to the specified size, the current number of pixels: {self.shape}",
                category=RuntimeWarning,
                stacklevel=2,
            )

        new_image = tp.empty(
            (self.__image.shape[0], shape[0], shape[1]), device=self.__image.device
        )

        for i, channel in enumerate(self.__image):
            self.shape = (self.__image.shape[0], shape[0], shape[1])
            left_border = tp.zeros((current_rows, left), device=self.__image.device)
            right_border = tp.zeros((current_rows, right), device=self.__image.device)
            center = tp.hstack((left_border, channel, right_border))

            top_center = tp.zeros((top, current_cols), device=self.__image.device)
            bottom_center = tp.zeros((bottom, current_cols), device=self.__image.device)

            top_left = tp.zeros((left, top), device=self.__image.device)
            top_right = tp.zeros((right, top), device=self.__image.device)
            bottom_left = tp.zeros((left, bottom), device=self.__image.device)
            bottom_right = tp.zeros((right, bottom), device=self.__image.device)

            top_border = tp.hstack((top_left, top_center, top_right))
            bottom_border = tp.hstack((bottom_left, bottom_center, bottom_right))

            new_image[i] = tp.vstack((top_border, center, bottom_border))

        self.__image = new_image

    def imshow(
        self, parts: list = ["intensity", "amplitude", "phase"], cmap: str = "jet"
    ) -> None:
        """
        Displays the current value of the intensity, amplitude, and phase of the image
        Parameters
        ----------
        `parts` : list - list of parameters to be displayed. Possible parameters: 'intensity', 'amplitude', 'phase'\n
        `cmap` : str - type of colormap used when displaying parts of the image. To see a list of available colormaps, use:
        >>> import matploltib.pyplot as plt
        >>> print(plt.colormaps)
        """
        if len(self.shape) == 2:
            num = len(parts)
            for i, part in enumerate(parts):
                plot_number = int(f"1{num}{i + 1}")

                plt.subplot(plot_number)
                plt.axis("off")
                plt.title(part)

                if part == "intensity":
                    plt.imshow(self.get_intensity().cpu().detach(), cmap=cmap)
                elif part == "amplitude":
                    plt.imshow(self.get_amplitude().cpu().detach(), cmap=cmap)
                elif part == "phase":
                    plt.imshow(self.get_phase().cpu().detach(), cmap=cmap)
                else:
                    raise SyntaxError(
                        "parts must be: 'intensity' or 'amplitude' or 'phase'"
                    )

        else:
            num = len(parts)
            for i, part in enumerate(parts):
                plot_number = int(f"1{num}{i + 1}")

                plt.subplot(plot_number)
                plt.axis("off")
                plt.title(part)

                if part == "intensity":
                    plt.imshow(self.get_intensity().cpu().detach()[0], cmap=cmap)
                elif part == "amplitude":
                    plt.imshow(self.get_amplitude().cpu().detach()[0], cmap=cmap)
                elif part == "phase":
                    plt.imshow(self.get_phase().cpu().detach()[0], cmap=cmap)
                else:
                    raise SyntaxError(
                        "parts must be: 'intensity' or 'amplitude' or 'phase'"
                    )
        plt.show()

    def imwrite(self, path: str, cmap: str = "jet") -> None: ...  ### change


class Lens:
    def __init__(self, size: float, shape: tuple) -> None:
        self.size = size
        self.shape = shape

    def __str__(self) -> str:
        return f"Lens(\n\tpixel width: {self.get_pixel_width()}m\n\tsize: {self.size}m\n\tshape: {self.shape}pixels\n)"

    def get_size(self) -> float:
        """
        Returns the current lens size in meters [m]
        """
        return self.size

    def get_shape(self) -> tuple:
        """
        Returns the current lens size in pixels [pixels]
        """
        return self.shape

    def get_pixel_width(self) -> float:
        """
        Returns the current width of the pixel in meters [m]
        """
        return self.size / self.shape[0]

    def set_pixel_width(self, width: float) -> None:
        """
        Sets the pixel width of the current image in meters [m]\n
        Parameters
        ----------
        `width` : float - pixel width in meters [m]
        """
        self.size = self.shape[0] * width

    def __add__(self): ...
    def __mul__(self): ...

    def get_amplitude(self): ...
    def get_phase(self): ...
    def get_real_part(self): ...
    def get_intensity(self): ...

    def set_amplitude(self): ...
    def set_phase(self): ...

    def imshow(self): ...


class TorchLens(Lens):
    def __init__(
        self, size: float, shape: tuple, lens: Union[tp.Tensor, None] = None
    ) -> None:
        super().__init__(size, shape)
        if lens is None:
            self.__lens = tp.rand(shape)
        else:
            self.__lens = lens

    def __add__(self, other):
        if isinstance(other, TorchOptImage):
            if (
                other.get_shape()[-1] != self.get_shape()[-1]
                or other.get_shape()[-2] != self.get_shape()[-2]
            ):
                raise ValueError(
                    f"it is impossible to make an addition: the size of the parts to be folded must be the same, but: ({other.shape[-2]}, {other.shape[-1]}) and ({self.get_shape()[-2]}, {self.get_shape()[-1]})"
                )

            other_phase = self.get_phase()
            current_phase = other.get_phase()
            current_amplitude = other.get_amplitude()
            output = current_amplitude * tp.exp(1j * (current_phase + other_phase))
            return TorchOptImage(output, other.get_wave_length(), other.get_size())
        else:
            raise SyntaxError(
                f"the second operand type is required to be a TorchOptImage, but: {type(other)}"
            )

    def __mul__(self, other):
        if isinstance(other, TorchOptImage):
            if (
                other.get_shape()[-1] != self.get_shape()[-1]
                or other.get_shape()[-2] != self.get_shape()[-2]
            ):
                raise ValueError(
                    f"it is impossible to make a multyply: the size of the parts to be folded must be the same, but: {other.get_shape()} and {self.get_shape()}"
                )

            current_phase = other.get_phase()
            other_amplitude = self.get_amplitude()
            current_amplitude = other.get_amplitude()
            output = (current_amplitude * other_amplitude) * tp.exp(
                1j * (current_phase)
            )
            return TorchOptImage(output, other.get_wave_length(), other.get_size())
        else:
            raise SyntaxError(
                f"the second operand type is required to be a TorchOptImage, but: {type(other)}"
            )

    def get_amplitude(self) -> tp.Tensor:
        """
        Returns the current value of the lens amplitude
        """
        return self.__lens.abs()

    def get_phase(self) -> tp.Tensor:
        """
        Returns the current value of the lens phase
        """
        return self.__lens.angle()

    def get_real_part(self) -> tp.Tensor:
        """
        Returns the current value of the real part of the lens
        """
        return self.__lens.real

    def get_intensity(self) -> tp.Tensor:
        """
        Returns the current value of the lens intensity
        """
        return self.get_real_part() ** 2

    def set_amplitude(self, src: tp.Tensor) -> None:
        """
        Changes the amplitude of the current lens
        Parameters
        ----------
        `phase` : Tensor - new image amplitude lens
        """
        current_phase = self.get_phase()
        self.__lens = src * tp.exp(1j * current_phase)

    def set_phase(self, src: tp.Tensor) -> None:
        """
        Changes the phase of the current lens
        Parameters
        ----------
        `phase` : Tensor - new image phase lens
        """
        current_amplitude = self.get_amplitude()
        self.__lens = current_amplitude * tp.exp(1j * src)

    def imshow(
        self, parts: list = ["intensity", "amplitude", "phase"], cmap: str = "jet"
    ) -> None:
        """
        Displays the current value of the intensity, amplitude, and phase of the lens
        Parameters
        ----------
        `parts` : list - list of parameters to be displayed. Possible parameters: 'intensity', 'amplitude', 'phase'\n
        `cmap` : str - type of colormap used when displaying parts of the lens. To see a list of available colormaps, use:
        >>> import matploltib.pyplot as plt
        >>> print(plt.colormaps)
        """
        num = len(parts)
        for i, part in enumerate(parts):
            plot_number = int(f"1{num}{i + 1}")

            plt.subplot(plot_number)
            plt.axis("off")
            plt.title(part)

            if part == "intensity":
                plt.imshow(self.get_intensity().cpu().detach(), cmap=cmap)
            elif part == "amplitude":
                plt.imshow(self.get_amplitude().cpu().detach(), cmap=cmap)
            elif part == "phase":
                plt.imshow(self.get_phase().cpu().detach(), cmap=cmap)
            else:
                raise SyntaxError(
                    "parts must be: 'intensity' or 'amplitude' or 'phase'"
                )
        plt.show()
