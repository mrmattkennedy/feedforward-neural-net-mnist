#include <WindowsConstants.au3>
#include <Array.au3>
#include <Misc.au3>
#include <ButtonConstants.au3>
#include <GUIConstantsEx.au3>
#include <StaticConstants.au3>
Opt("GUIOnEventMode", True)
Opt("GUICloseOnEsc", False)

Global $gui = GUICreate("Sigmoid", 140, 100, -500, 100)
	GUISetOnEvent($GUI_EVENT_CLOSE, "_Exit")

Global $input = GUICtrlCreateInput("", 5, 5, 130, 20)
Global $output = GUICtrlCreateLabel("", 5, 30, 130, 30, BitOR($SS_CENTER, $SS_CENTERIMAGE), BitOR($SS_BLACKFRAME, $SS_CENTERIMAGE))
Global $calc = GUICtrlCreateButton("Calc", 5, 65, 65, 30)
	GUICtrlSetOnEvent(-1, "Sigmoid")
Global $exit = GUICtrlCreateButton("Exit", 70, 65, 65, 30)
	GUICtrlSetOnEvent(-1, "_Exit")

GUISetState(@SW_SHOW, $gui)

While 1
	Sleep(20)
WEnd

Func Sigmoid()
	Local $sigmoid_input = GUICtrlRead($input)
	If Not StringIsDigit($sigmoid_input) And Not StringIsFloat($sigmoid_input) Then
		GUICtrlSetData($output, "NAN")
		Return
	EndIf

	If $sigmoid_input > 88 Then $sigmoid_input = 88
	If $sigmoid_input < -88 Then $sigmoid_input = -88
	Local $sigmoid = Round(1 / (1 + Exp(-$sigmoid_input)), 8)

	If @error == 0 And $input <> '' Then
		GUICtrlSetData($output, "Sigmoid is " & $sigmoid)
	EndIf

EndFunc

Func _Exit()
	Exit
EndFunc