import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import React from "react";
import HomeChatScreen from "./HomeChatScreen";
import FriendsScreen from "./FriendsScreen";
import ChatsScreen from "./ChatsScreen";
import ChatMessagesScreen from "./ChatMessagesScreen";
import AIChatScreen from './AiChat/AIChatScreen'
const Stack = createNativeStackNavigator();

const StackNavigator = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="HomeChatScreen" component={HomeChatScreen} />
        <Stack.Screen name="FriendsScreen" component={FriendsScreen} />
        <Stack.Screen name="ChatsScreen" component={ChatsScreen} />
        <Stack.Screen name='AIChatScreen' component={AIChatScreen}/>
        <Stack.Screen
          name="ChatMessagesScreen"
          component={ChatMessagesScreen}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default StackNavigator;
