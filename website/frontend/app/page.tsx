'use client';

import { useRouter } from 'next/navigation';
import { Button, Form, Input, FloatButton, Tooltip } from "antd";
import { LockOutlined, UserOutlined, QuestionCircleOutlined } from '@ant-design/icons';

type LoginFormValues = {
  username: string;
  password: string;
}

export default function Home() {
  const router = useRouter();

  const onFinish = (values: LoginFormValues) => {
    fetch('http://localhost:8000/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify(values)
    })
    .then(response => response.json())
    .then(data => {
      if (data.status) {
        router.push('/dashboard');
      } else {
        console.log('Ошибка логина');
      }
    })
    .catch(error => {
      console.error('Ошибка', error);
    });
  }

  return (
    <div className="h-screen flex justify-center items-center">
      <Form
        name="login"
        initialValues={{ remember: true }}
        style={{ minWidth: 400 }}
        onFinish={onFinish}
      >
        <Form.Item
          name="username"
          rules={[{ required: true, message: "Пожалуйста, введите имя пользователя." }]}
        >
          <Input prefix={<UserOutlined />} placeholder="Имя пользователя" />
        </Form.Item>
        <Form.Item
          name="password"
          rules={[{ required: true, message: "Пожалуйста, введите пароль." }]}
        >
          <Input.Password prefix={<LockOutlined />} placeholder="Пароль" />
        </Form.Item>
        <Form.Item>
          <Button block type="primary" htmlType="submit">
            Войти
          </Button>
        </Form.Item>
      </Form>

      <Tooltip title="Если вы забыли пароль или не зарегистрированы, обратитесь к системному администратору.">
        <FloatButton
          icon={<QuestionCircleOutlined />}
          type="primary"
          style={{ insetInlineEnd: 24 }}
        />
      </Tooltip>
    </div>
  );
}
