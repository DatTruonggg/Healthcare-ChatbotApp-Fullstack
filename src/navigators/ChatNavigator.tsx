import { StyleSheet, Text, View, StatusBar } from "react-native";
import StackNavigator from "../screens/chat/StackNavigator";
import { UserContext } from "../screens/chat/components/UserContext";
import React from "react";

export default function ChatNavigator() {
	return (
		<>
			<UserContext>
				<StackNavigator />
			</UserContext>
		</>
	);
}

const styles = StyleSheet.create({
	container: {
		flex: 1,
		backgroundColor: "#fff",
		alignItems: "center",
		justifyContent: "center",
	},
});
