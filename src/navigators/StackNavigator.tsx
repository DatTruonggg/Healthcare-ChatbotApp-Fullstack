import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import HomeChatScreen from "../screens/chat/HomeChatScreen";
import FriendsScreen from "../screens/chat/FriendsScreen";
import ChatsScreen from "../screens/chat/ChatsScreen";
import ChatMessagesScreen from "../screens/chat/ChatMessagesScreen";

const Stack = createNativeStackNavigator();

const StackNavigator = () => {
	return (
		<NavigationContainer>
			<Stack.Navigator>
				<Stack.Screen name="HomeChatScreen" component={HomeChatScreen} />
				<Stack.Screen name="FriendsScreen" component={FriendsScreen} />
				<Stack.Screen name="ChatsScreen" component={ChatsScreen} />
				<Stack.Screen name="ChatMessagesScreen" component={ChatMessagesScreen} />
			</Stack.Navigator>
		</NavigationContainer>
	);
};

export default StackNavigator;
