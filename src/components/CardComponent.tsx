import { View, Text, TouchableOpacity, StyleProp, ViewStyle } from 'react-native';
import React, { ReactNode } from 'react';
import { globalStyles } from '../styles/globalStyles';
import { appColors } from '../constants/appColors';

interface Props {
  onPress?: () => void;
  children: ReactNode;
  styles?: StyleProp<ViewStyle>;
  isShadow?: boolean;
  color?: string
}

const CardComponent = (props: Props) => {
  const { children, isShadow, onPress, color, styles } = props;
  const localStyle: StyleProp<ViewStyle>[] = [
    globalStyles.card,
    isShadow ? globalStyles.shadow : undefined,
    { backgroundColor: color ?? appColors.white },
    styles,
  ];
  return (
    <TouchableOpacity style={localStyle} onPress={onPress}>
      {children}
    </TouchableOpacity>
  );
};

export default CardComponent;
