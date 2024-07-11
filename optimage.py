from warnings import warn

import numpy as np
import cv2
import matplotlib.pyplot as plt


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

    def append_amplitude(self): ...
    def append_phase(self): ...

    def get_amplitude(self): ...
    def get_phase(self): ...
    def get_real_part(self): ...
    def get_intensive(self): ...

    def fresnell(self): ...
    def pad_zeros(self): ...
    def imshow(self): ...
    def imwrite(self): ...


class NumpyOptImage(OptImage):
    def __init__(self, image: np.ndarray, wave_length: float, size: float) -> None:
        super().__init__(wave_length, size, image.shape)
        self.__image = image

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

    def get_intensive(self) -> np.ndarray:
        """
        Returns the current value of the image intensive
        """
        return self.get_real_part() ** 2

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
        self, parts: list = ["intensive", "amplitude", "phase"], cmap: str = "jet"
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

            if part == "intensive":
                plt.imshow(self.get_intensive(), cmap=cmap)
            elif part == "amplitude":
                plt.imshow(self.get_amplitude(), cmap=cmap)
            elif part == "phase":
                plt.imshow(self.get_phase(), cmap=cmap)
            else:
                raise SyntaxError(
                    "parts must be: 'intensive' or 'amplitude' or 'phase'"
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
        parts: list = ["intensive", "amplitude", "phase"]
        num = len(parts)
        for i, part in enumerate(parts):
            plot_number = int(f"1{num}{i + 1}")

            plt.subplot(plot_number)
            plt.axis("off")
            plt.title(part)

            if part == "intensive":
                plt.imshow(self.get_intensive(), cmap=cmap)
            elif part == "amplitude":
                plt.imshow(self.get_amplitude(), cmap=cmap)
            elif part == "phase":
                plt.imshow(self.get_phase(), cmap=cmap)
        plt.savefig(path)
