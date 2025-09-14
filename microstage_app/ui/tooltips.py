from __future__ import annotations

from PySide6 import QtWidgets, QtCore


class _TooltipContextFilter(QtCore.QObject):
    """Event filter showing a widget's tooltip on context menu events."""

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # type: ignore[override]
        if event.type() == QtCore.QEvent.ContextMenu:
            tip = obj.toolTip() if hasattr(obj, "toolTip") else ""
            if tip:
                QtWidgets.QToolTip.showText(event.globalPos(), tip, obj)  # type: ignore[arg-type]
                return True
        return super().eventFilter(obj, event)


def install_tooltip_context(widget: QtWidgets.QWidget) -> None:
    """Install context-menu tooltip behavior on *widget*."""
    filt = _TooltipContextFilter(widget)
    widget.installEventFilter(filt)
    # Store reference to avoid garbage collection
    widget._tooltip_ctx_filter = filt  # type: ignore[attr-defined]


_INPUT_WIDGET_TYPES = (
    QtWidgets.QLineEdit,
    QtWidgets.QComboBox,
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QSlider,
    QtWidgets.QAbstractButton,
)


def apply_tooltip_context(root: QtWidgets.QWidget) -> None:
    """Apply tooltip context behavior to all input widgets under *root*."""
    for t in _INPUT_WIDGET_TYPES:
        for w in root.findChildren(t):
            install_tooltip_context(w)
